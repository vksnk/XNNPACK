// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::AllOf;

// The shape of an attention score computation for one query token: `q` is
// [batch, heads, queries, head] and `k` is [batch, heads, keys, head], and the
// scores are `q` times `k` transposed.
struct ScoresGraph {
  static constexpr size_t kBatch = 1;
  static constexpr size_t kHeads = 2;
  static constexpr size_t kHead = 8;

  uint32_t q_id = 0;
  uint32_t k_id = 1;
  uint32_t scores_id = 2;
  SubgraphBuilder builder{3};

  // Builds scores = dot(q, transpose(k)). `queries` == 1 is the decode case
  // the rewrite is for.
  ScoresGraph(size_t queries, size_t keys) {
    builder.AddInput(ynn_type_fp32, {kBatch, kHeads, queries, kHead}, q_id)
        .AddInput(ynn_type_fp32, {kBatch, kHeads, keys, kHead}, k_id)
        .AddOutput(ynn_type_fp32, {kBatch, kHeads, queries, keys}, scores_id);

    const int32_t perm[] = {0, 1, 3, 2};
    uint32_t k_transposed_id = YNN_INVALID_VALUE_ID;
    EXPECT_EQ(ynn_define_static_transpose(builder.GetSubgraph(), /*rank=*/4,
                                          perm, k_id, &k_transposed_id,
                                          /*flags=*/0),
              ynn_status_success);

    uint32_t out_id = scores_id;
    EXPECT_EQ(ynn_define_dot(builder.GetSubgraph(), /*num_k_dims=*/1, q_id,
                             k_transposed_id,
                             /*input_c_id=*/YNN_INVALID_VALUE_ID, &out_id,
                             /*flags=*/0),
              ynn_status_success);
  }
};

// Returns the operand the dot reduces over rows of, i.e. the one that is used
// in its natural layout and never packed.
uint32_t DotRowOperand(const ynn_subgraph& subgraph) {
  for (const ynn_node& node : subgraph.nodes) {
    if (!node.is_valid()) continue;
    if (std::holds_alternative<ynn_node::dot>(node.op)) return node.inputs[0];
  }
  return YNN_INVALID_VALUE_ID;
}

// Returns true if any value derived from `id` is packed, which is the copy of
// the whole operand we are trying to avoid.
bool IsPacked(const ynn_subgraph& subgraph, uint32_t id) {
  std::vector<uint32_t> derived = {id};
  for (const ynn_node& node : subgraph.nodes) {
    if (!node.is_valid() || node.inputs.empty()) continue;
    const bool from_id = std::find(derived.begin(), derived.end(),
                                   node.inputs[0]) != derived.end();
    if (!from_id) continue;
    if (std::holds_alternative<ynn_node::pack_b>(node.op)) return true;
    if (!node.outputs.empty()) derived.push_back(node.outputs[0]);
  }
  return false;
}

TEST(fusion, dot_of_transpose_single_query) {
  ScoresGraph graph(/*queries=*/1, /*keys=*/64);
  ynn_subgraph& subgraph = *graph.builder.GetSubgraph();
  ASSERT_EQ(DotRowOperand(subgraph), graph.q_id);
  ASSERT_TRUE(IsPacked(subgraph, graph.k_id));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // The keys are now the operand used in its natural layout, so they are
  // neither transposed nor packed; only the single query row is.
  EXPECT_EQ(DotRowOperand(subgraph), graph.k_id);
  EXPECT_FALSE(IsPacked(subgraph, graph.k_id));
  EXPECT_TRUE(IsPacked(subgraph, graph.q_id));
  // The result is put back into the layout the consumers expect.
  EXPECT_THAT(ProducerOf(graph.scores_id, subgraph), IsStaticTranspose());
}

TEST(fusion, dot_of_transpose_not_applied_to_many_queries) {
  // With more than one query row the swap would need real transposes, and the
  // dot would lose its wide dimension, so the graph must be left alone.
  ScoresGraph graph(/*queries=*/4, /*keys=*/64);
  ynn_subgraph& subgraph = *graph.builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_EQ(DotRowOperand(subgraph), graph.q_id);
  EXPECT_THAT(ProducerOf(graph.scores_id, subgraph), IsDot());
}

// The rewritten graph has to compute the same scores.
TEST(fusion, dot_of_transpose_is_correct) {
  constexpr size_t kKeys = 37;  // Not a multiple of any vector width.
  ScoresGraph graph(/*queries=*/1, /*keys=*/kKeys);

  Runtime runtime(graph.builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const std::vector<size_t> q_shape = {ScoresGraph::kBatch, ScoresGraph::kHeads,
                                       1, ScoresGraph::kHead};
  const std::vector<size_t> k_shape = {ScoresGraph::kBatch, ScoresGraph::kHeads,
                                       kKeys, ScoresGraph::kHead};
  const std::vector<size_t> scores_shape = {ScoresGraph::kBatch,
                                            ScoresGraph::kHeads, 1, kKeys};

  Tensor<float> q(q_shape);
  Tensor<float> k(k_shape);
  Tensor<float> scores(scores_shape);
  for (size_t h = 0; h < ScoresGraph::kHeads; ++h) {
    for (size_t c = 0; c < ScoresGraph::kHead; ++c) {
      q(0, h, 0, c) = 0.25f * static_cast<float>(c) - 1.0f + h;
    }
    for (size_t s = 0; s < kKeys; ++s) {
      for (size_t c = 0; c < ScoresGraph::kHead; ++c) {
        k(0, h, s, c) = 0.5f * static_cast<float>((s + c) % 5) - 1.0f;
      }
    }
  }

  runtime.ReshapeExternalTensor(q_shape, q.base(), graph.q_id)
      .ReshapeExternalTensor(k_shape, k.base(), graph.k_id)
      .ReshapeRuntime()
      .SetupExternalTensor(scores.base(), graph.scores_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t h = 0; h < ScoresGraph::kHeads; ++h) {
    for (size_t s = 0; s < kKeys; ++s) {
      float expected = 0.0f;
      for (size_t c = 0; c < ScoresGraph::kHead; ++c) {
        expected += q(0, h, 0, c) * k(0, h, s, c);
      }
      EXPECT_NEAR(scores(0, h, 0, s), expected, 1e-4f) << "h=" << h << " s=" << s;
    }
  }
}

// The same graph with the operand types a dynamically quantized attention has:
// the queries are unsigned and the keys are signed.
struct QuantizedScoresGraph {
  static constexpr size_t kHeads = 2;
  static constexpr size_t kHead = 8;

  uint32_t q_id = 0;
  uint32_t k_id = 1;
  uint32_t scores_id = 2;
  SubgraphBuilder builder{3};

  QuantizedScoresGraph(size_t queries, size_t keys, ynn_type k_type)
      : QuantizedScoresGraph(queries, keys, ynn_type_uint8, k_type) {}

  QuantizedScoresGraph(size_t queries, size_t keys, ynn_type q_type,
                       ynn_type k_type) {
    builder.AddInput(q_type, {kHeads, queries, kHead}, q_id)
        .AddInput(k_type, {kHeads, keys, kHead}, k_id)
        .AddOutput(ynn_type_int32, {kHeads, queries, keys}, scores_id);

    const int32_t perm[] = {0, 2, 1};
    uint32_t k_transposed_id = YNN_INVALID_VALUE_ID;
    EXPECT_EQ(ynn_define_static_transpose(builder.GetSubgraph(), /*rank=*/3,
                                          perm, k_id, &k_transposed_id,
                                          /*flags=*/0),
              ynn_status_success);

    uint32_t out_id = scores_id;
    EXPECT_EQ(ynn_define_dot(builder.GetSubgraph(), /*num_k_dims=*/1, q_id,
                             k_transposed_id,
                             /*input_c_id=*/YNN_INVALID_VALUE_ID, &out_id,
                             /*flags=*/0),
              ynn_status_success);
  }
};

TEST(fusion, dot_of_transpose_quantized) {
  // A key cache is packed on every decode step, which is the whole cost the
  // rewrite exists to remove, and it is quantized in every real model.
  QuantizedScoresGraph graph(/*queries=*/1, /*keys=*/64, ynn_type_int8);
  ynn_subgraph& subgraph = *graph.builder.GetSubgraph();
  ASSERT_TRUE(IsPacked(subgraph, graph.k_id));

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_EQ(DotRowOperand(subgraph), graph.k_id);
  EXPECT_FALSE(IsPacked(subgraph, graph.k_id));
  EXPECT_TRUE(IsPacked(subgraph, graph.q_id));
}

TEST(fusion, dot_of_transpose_declines_without_a_kernel_for_swapped_types) {
  // Swapping uint8 x int4 would ask for an int4 x uint8 kernel, which does not
  // exist: the sub-byte operand only has kernels as the packed one.
  QuantizedScoresGraph graph(/*queries=*/1, /*keys=*/64, ynn_type_int4);
  ynn_subgraph& subgraph = *graph.builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  EXPECT_EQ(DotRowOperand(subgraph), graph.q_id);
}

TEST(fusion, dot_of_transpose_quantized_is_correct) {
  constexpr size_t kKeys = 37;  // Not a multiple of any vector width.
  QuantizedScoresGraph graph(/*queries=*/1, /*keys=*/kKeys, ynn_type_int8);

  Runtime runtime(graph.builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const std::vector<size_t> q_shape = {QuantizedScoresGraph::kHeads, 1,
                                       QuantizedScoresGraph::kHead};
  const std::vector<size_t> k_shape = {QuantizedScoresGraph::kHeads, kKeys,
                                       QuantizedScoresGraph::kHead};
  const std::vector<size_t> scores_shape = {QuantizedScoresGraph::kHeads, 1,
                                            kKeys};

  Tensor<uint8_t> q(q_shape);
  Tensor<int8_t> k(k_shape);
  Tensor<int32_t> scores(scores_shape);
  // Values at both ends of each range, so a kernel that got the signedness of
  // an operand wrong could not pass.
  for (size_t h = 0; h < QuantizedScoresGraph::kHeads; ++h) {
    for (size_t c = 0; c < QuantizedScoresGraph::kHead; ++c) {
      q(h, 0, c) = static_cast<uint8_t>(200 + c + h);
    }
    for (size_t s = 0; s < kKeys; ++s) {
      for (size_t c = 0; c < QuantizedScoresGraph::kHead; ++c) {
        k(h, s, c) = static_cast<int8_t>(-128 + ((s * 7 + c * 13) % 256));
      }
    }
  }

  runtime.ReshapeExternalTensor(q_shape, q.base(), graph.q_id)
      .ReshapeExternalTensor(k_shape, k.base(), graph.k_id)
      .ReshapeRuntime()
      .SetupExternalTensor(scores.base(), graph.scores_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t h = 0; h < QuantizedScoresGraph::kHeads; ++h) {
    for (size_t s = 0; s < kKeys; ++s) {
      int32_t expected = 0;
      for (size_t c = 0; c < QuantizedScoresGraph::kHead; ++c) {
        expected += static_cast<int32_t>(q(h, 0, c)) * k(h, s, c);
      }
      EXPECT_EQ(scores(h, 0, s), expected) << "h=" << h << " s=" << s;
    }
  }
}

}  // namespace
}  // namespace ynn
