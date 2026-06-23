// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Numerically validates the flash-attention scheduling. The pattern
// QK^T -> softmax -> P*V is fused into a shared query-tile loop (so the score
// and probability matrices are never fully materialized) without changing the
// result; the output must match the reference.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

// out[bh,m,p] = sum_n softmax_n(sum_k q[bh,m,k]*key[k,n]) * v[n,p]
Tensor<float> ReferenceAttention(const Tensor<float>& q, const Tensor<float>& key,
                                 const Tensor<float>& v) {
  const size_t BH = q.extent(0), M = q.extent(1), K = q.extent(2);
  const size_t N = key.extent(1), P = v.extent(1);
  Tensor<float> out({BH, M, P});
  for (size_t bh = 0; bh < BH; ++bh) {
    for (size_t m = 0; m < M; ++m) {
      std::vector<float> s(N);
      float max = -std::numeric_limits<float>::infinity();
      for (size_t n = 0; n < N; ++n) {
        float acc = 0;
        for (size_t k = 0; k < K; ++k) acc += q(bh, m, k) * key(k, n);
        s[n] = acc;
        max = std::max(max, acc);
      }
      float sum = 0;
      for (size_t n = 0; n < N; ++n) {
        s[n] = std::exp(s[n] - max);
        sum += s[n];
      }
      for (size_t p = 0; p < P; ++p) {
        float acc = 0;
        for (size_t n = 0; n < N; ++n) acc += (s[n] / sum) * v(n, p);
        out(bh, m, p) = acc;
      }
    }
  }
  return out;
}

TEST(FlashAttention, MatchesReference) {
  ReplicableRandomDevice rng;
  // Query count exceeds the flash query block, so queries are actually tiled.
  const size_t BH = 3, M = 200, K = 64, N = 128, P = 48;
  const std::vector<size_t> q_shape = {BH, M, K};
  const std::vector<size_t> key_shape = {K, N};
  const std::vector<size_t> v_shape = {N, P};
  const std::vector<size_t> scores_shape = {BH, M, N};
  const std::vector<size_t> scores_reduced = {BH, M, 1};
  const std::vector<size_t> out_shape = {BH, M, P};

  SubgraphBuilder subgraph(13);
  uint32_t q_id = 0, key_id = 1, v_id = 2, scores_id = 3, max_id = 4,
           sub_id = 5, exp_id = 6, sum_id = 7, inv_id = 8, probs_id = 9,
           out_id = 10, one_id = 11;

  subgraph.AddInput(ynn_type_fp32, q_shape, q_id)
      .AddInput(ynn_type_fp32, key_shape, key_id)
      .AddInput(ynn_type_fp32, v_shape, v_id)
      .AddTensor(ynn_type_fp32, scores_shape, scores_id)
      .AddTensor(ynn_type_fp32, scores_reduced, max_id)
      .AddTensor(ynn_type_fp32, scores_shape, sub_id)
      .AddTensor(ynn_type_fp32, scores_shape, exp_id)
      .AddTensor(ynn_type_fp32, scores_reduced, sum_id)
      .AddTensor(ynn_type_fp32, scores_reduced, inv_id)
      .AddTensor(ynn_type_fp32, scores_shape, probs_id)
      .AddOutput(ynn_type_fp32, out_shape, out_id)
      .AddScalar(1.0f, one_id);

  // scores = Q . K
  subgraph.AddDot(/*num_k_dims=*/1, q_id, key_id, YNN_INVALID_VALUE_ID,
                  scores_id);
  // softmax(scores) over the last axis, ending in multiply (as the real
  // softmax lowering does) so the P*V matmul fuses too.
  subgraph.AddReduce(ynn_reduce_max, {2}, scores_id, YNN_INVALID_VALUE_ID,
                     max_id, YNN_NODE_FLAG_KEEP_DIMS);
  subgraph.AddBinary(ynn_binary_subtract, scores_id, max_id, sub_id);
  subgraph.AddUnary(ynn_unary_exp, sub_id, exp_id);
  subgraph.AddReduce(ynn_reduce_sum, {2}, exp_id, YNN_INVALID_VALUE_ID, sum_id,
                     YNN_NODE_FLAG_KEEP_DIMS);
  subgraph.AddBinary(ynn_binary_divide, one_id, sum_id, inv_id);
  subgraph.AddBinary(ynn_binary_multiply, exp_id, inv_id, probs_id);
  // out = probs . V
  subgraph.AddDot(/*num_k_dims=*/1, probs_id, v_id, YNN_INVALID_VALUE_ID,
                  out_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> q(q_shape), key(key_shape), v(v_shape);
  fill_random(q.data(), q.size(), rng, -1.0f, 1.0f);
  fill_random(key.data(), key.size(), rng, -1.0f, 1.0f);
  fill_random(v.data(), v.size(), rng, -1.0f, 1.0f);

  Tensor<float> out(out_shape);
  runtime.ReshapeExternalTensor(q_shape, q.data(), q_id)
      .ReshapeExternalTensor(key_shape, key.data(), key_id)
      .ReshapeExternalTensor(v_shape, v.data(), v_id);
  runtime.SetupExternalTensor(out.data(), out_id);
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> expected = ReferenceAttention(q, key, v);
  for (size_t bh = 0; bh < BH; ++bh) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t p = 0; p < P; ++p) {
        ASSERT_NEAR(out(bh, m, p), expected(bh, m, p), 1e-4)
            << "at [" << bh << "," << m << "," << p << "]";
      }
    }
  }
}

}  // namespace
}  // namespace ynn
