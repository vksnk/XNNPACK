// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include "slinky/builder/simplify.h"

namespace ynn {
namespace {

bool contains(const std::string& str, const std::string& substr) {
  return std::search(str.begin(), str.end(), substr.begin(), substr.end()) !=
         str.end();
}

template <typename AT, typename BT>
void VerifyDotLoopOrder(const std::vector<size_t>& a_shape,
                        const std::vector<size_t>& b_shape,
                        bool expect_split_k, bool computed_a = false,
                        bool computed_b = false) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<AT>(), a_shape, a_id)
      .AddInput(type_of<BT>(), b_shape, b_id)
      .AddOutput(type_of<float>(), TensorShape({a_shape[0], b_shape[1]}),
                 out_id);
  uint32_t dot_a = a_id, dot_b = b_id;
  if (computed_a) {
    dot_a = YNN_INVALID_VALUE_ID;
    builder.AddTensor(type_of<AT>(), a_shape, dot_a)
        .AddUnary(ynn_unary_abs, a_id, dot_a);
  }
  if (computed_b) {
    dot_b = YNN_INVALID_VALUE_ID;
    builder.AddTensor(type_of<BT>(), b_shape, dot_b)
        .AddUnary(ynn_unary_abs, b_id, dot_b);
  }
  builder.AddDot(1, dot_a, dot_b, YNN_INVALID_VALUE_ID, out_id);

  TestScheduler scheduler(3);
  Runtime runtime(builder.GetSubgraph(), &scheduler,
                  YNN_FLAG_ENABLE_SLINKY_TRACE);

  std::vector<std::string> trace_events;
  std::mutex trace_mutex;  // NOLINT(build/c++11)
  runtime.get()->eval_config.trace_begin =
      [&](const char* name) -> slinky::index_t {
    std::lock_guard<std::mutex> lock(trace_mutex);  // NOLINT(build/c++11)
    trace_events.push_back(name);
    return 0;
  };

  Tensor<AT> a(a_shape);
  Tensor<BT> b(b_shape);
  Tensor<float> out({a_shape[0], b_shape[1]});
  a.fill(AT(-1.0f));
  b.fill(BT(-2.0f));
  runtime.ReshapeExternalTensor(a.extents(), a.data(), a_id)
      .ReshapeExternalTensor(b.extents(), b.data(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out.data(), out_id)
      .InvokeRuntime();
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  const float expected = (computed_a == computed_b ? 2.0f : -2.0f) * a_shape[1];
  for (size_t i = 0; i < a_shape[0]; ++i) {
    for (size_t j = 0; j < b_shape[1]; ++j) {
      ASSERT_EQ(out({i, j}), expected) << i << " " << j;
    }
  }

  EXPECT_FALSE(trace_events.empty());
  EXPECT_EQ(trace_events.front(), "pipeline");
  bool found_pack = false;
  bool found_dot = false;
  std::string first_loop;
  for (const std::string& event : trace_events) {
    if (contains(event, "pack_b")) found_pack = true;
    if (contains(event, "dot")) found_dot = true;
    if (first_loop.empty() &&
        (contains(event, "loop k") || contains(event, "loop d")) &&
        !contains(event, "iteration")) {
      first_loop = event;
    }
  }
  EXPECT_TRUE(found_pack);
  EXPECT_TRUE(found_dot);
  if (expect_split_k) {
    EXPECT_THAT(first_loop, testing::HasSubstr("loop k"));
  } else {
    EXPECT_THAT(first_loop, testing::Not(testing::HasSubstr("loop k")));
  }
}

TEST(DotSchedulingTest, NoSplitK) {
  VerifyDotLoopOrder<float, float>({300, 100}, {100, 400}, false);
}

TEST(DotSchedulingTest, SplitKTrue) {
  VerifyDotLoopOrder<float, float>({300, 8192}, {8192, 400}, true);
}

TEST(DotSchedulingTest, NarrowTypeNoSplitK) {
  VerifyDotLoopOrder<bfloat16, bfloat16>({300, 8192}, {8192, 400}, false);
}

TEST(DotSchedulingTest, NarrowTypeLargeSplitK) {
  VerifyDotLoopOrder<bfloat16, bfloat16>({300, 16384}, {16384, 400}, true);
}

// A computed input should stream by rows even when the dot splits K.
TEST(DotSchedulingTest, ComputedAStreamsRows) {
  VerifyDotLoopOrder<float, float>({300, 8192}, {8192, 400}, false, true);
}

TEST(DotSchedulingTest, ComputedBStreamsColumns) {
  VerifyDotLoopOrder<float, float>({300, 8192}, {8192, 400}, false, false, true);
}

TEST(DotSchedulingTest, TwoComputedInputsPreserveSplitK) {
  VerifyDotLoopOrder<float, float>({300, 8192}, {8192, 400}, true, true, true);
}

}  // namespace
}  // namespace ynn
