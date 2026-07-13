// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Tests for the automatic scheduler's loop fusion decisions. These tests
// build small subgraphs and inspect the loop structure of the resulting
// slinky pipeline to check which functions ended up sharing loops.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/print.h"
#include "slinky/runtime/stmt.h"

namespace ynn {
namespace {

// Records, for every call_stmt in the pipeline body, the stack of loops
// enclosing it, keyed by the name of the call.
class call_loops_collector : public slinky::recursive_node_visitor {
 public:
  using loop_stack = std::vector<slinky::var>;

  void visit(const slinky::loop* op) override {
    loops_.push_back(op->sym);
    slinky::recursive_node_visitor::visit(op);
    loops_.pop_back();
  }

  void visit(const slinky::call_stmt* op) override {
    calls_[op->attrs.name].push_back(loops_);
  }

  // Returns the loop stacks of all calls whose name starts with `prefix`.
  std::vector<loop_stack> find(const std::string& prefix) const {
    std::vector<loop_stack> result;
    for (const auto& [name, stacks] : calls_) {
      if (name.rfind(prefix, 0) == 0) {
        result.insert(result.end(), stacks.begin(), stacks.end());
      }
    }
    return result;
  }

 private:
  loop_stack loops_;
  std::map<std::string, std::vector<loop_stack>> calls_;
};

// All loop stacks come from the same stmt tree, so the loops shared between
// two calls are the common prefix of their loop stacks.
size_t common_loops(const call_loops_collector::loop_stack& a,
                    const call_loops_collector::loop_stack& b) {
  size_t d = 0;
  while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
  return d;
}

// The maximum number of shared loops over all pairs of calls from `a` and `b`.
size_t max_common_loops(
    const std::vector<call_loops_collector::loop_stack>& a,
    const std::vector<call_loops_collector::loop_stack>& b) {
  size_t result = 0;
  for (const auto& i : a) {
    for (const auto& j : b) {
      result = std::max(result, common_loops(i, j));
    }
  }
  return result;
}

// Builds the runtime the same way `Runtime` in subgraph_builder.cc does, but
// keeps the raw ynn_runtime so tests can inspect the scheduled pipeline.
class LoopFusionTest : public testing::Test {
 protected:
  void MakeRuntime(ynn_subgraph_t subgraph) {
    ynn_threadpool_t threadpool = nullptr;
    ASSERT_EQ(ynn_create_threadpool(TestScheduler::scheduler(), &scheduler_,
                                    /*flags=*/0, &threadpool),
              ynn_status_success);
    threadpool_.reset(threadpool);
    ASSERT_EQ(ynn_optimize_subgraph(subgraph, threadpool, /*flags=*/0),
              ynn_status_success);
    ynn_runtime_t runtime = nullptr;
    ASSERT_EQ(ynn_create_runtime(subgraph, threadpool, /*flags=*/0, &runtime),
              ynn_status_success);
    runtime_.reset(runtime);
    runtime_->pipeline.body.accept(&calls_);
  }

  std::string PipelineToString() const {
    std::stringstream ss;
    ss << std::tuple<const slinky::stmt&, const slinky::node_context&>(
        runtime_->pipeline.body, runtime_->globals.symbols);
    return ss.str();
  }

  TestScheduler scheduler_{3};
  std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>
      threadpool_{nullptr, ynn_delete_threadpool};
  std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)> runtime_{
      nullptr, ynn_delete_runtime};
  call_loops_collector calls_;
};

// pack_b should be computed inside the dot's loop nest, so packing happens
// per-block instead of materializing the whole packed buffer up front.
TEST_F(LoopFusionTest, PackFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  const auto pack = calls_.find("pack_b");
  const auto dot = calls_.find("dot");
  ASSERT_FALSE(pack.empty());
  ASSERT_FALSE(dot.empty());
  EXPECT_GE(max_common_loops(pack, dot), 1) << PipelineToString();
}

// An elementwise producer of the packed input, i.e. dot(A, pack(exp(B))),
// should be fused into the same loop nest as the pack and the dot, so the
// intermediate buffer is computed per-block too.
TEST_F(LoopFusionTest, ProducerOfPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddDot(1, a_id, exp_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  const auto exp = calls_.find("exp");
  const auto pack = calls_.find("pack_b");
  const auto dot = calls_.find("dot");
  ASSERT_FALSE(exp.empty());
  ASSERT_FALSE(pack.empty());
  ASSERT_FALSE(dot.empty());
  // The pack itself fuses with the dot (via the scheduler_bound mechanism).
  EXPECT_GE(max_common_loops(pack, dot), 1) << PipelineToString();
  // The producer of the packed input should end up in the same loop nest,
  // but currently does not: the source region inference breaks at pack's
  // non-identity input bounds, so exp (and anything else feeding the pack)
  // is scheduled at the root.
  EXPECT_GE(max_common_loops(exp, dot), 1) << PipelineToString();
}

// Same as above, but B arrives transposed: dot(A, transpose(exp(Bt))). The
// transpose is folded into the packing (always_alias_transpose), so the func
// chain is exp -> transpose (aliased copy) -> pack_b -> dot. In this layout
// exp's loop order matches the dot's loop nest positionally, so fusion of exp
// is blocked *only* by the source region inference breaking at pack's
// non-identity input bounds - unlike the test above, which additionally
// requires order-insensitive split matching.
TEST_F(LoopFusionTest, ProducerOfTransposedPackedInputFusesWithDot) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  uint32_t transpose_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<float>(), TensorShape(2), a_id)
      .AddInput(type_of<float>(), TensorShape(2), b_id)
      .AddOutput(type_of<float>(), TensorShape(2), out_id)
      .AddTensor(type_of<float>(), TensorShape(2), exp_id)
      .AddTensor(type_of<float>(), TensorShape(2), transpose_id)
      .AddUnary(ynn_unary_exp, b_id, exp_id)
      .AddTranspose({1, 0}, exp_id, transpose_id)
      .AddDot(1, a_id, transpose_id, YNN_INVALID_VALUE_ID, out_id);

  MakeRuntime(subgraph.GetSubgraph());

  const auto exp = calls_.find("exp");
  const auto pack = calls_.find("pack_b");
  const auto dot = calls_.find("dot");
  ASSERT_FALSE(exp.empty());
  ASSERT_FALSE(pack.empty());
  ASSERT_FALSE(dot.empty());
  // The transpose should have been folded into the packing.
  EXPECT_TRUE(calls_.find("transpose").empty()) << PipelineToString();
  EXPECT_GE(max_common_loops(pack, dot), 1) << PipelineToString();
  EXPECT_GE(max_common_loops(exp, dot), 1) << PipelineToString();
}

}  // namespace
}  // namespace ynn
