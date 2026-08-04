// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Correctness test for the attention graphs in attention_graph.h, comparing
// both `define_attention` and `define_attention_decode1` against a naive CPU
// reference. Previously only attention_bench built these graphs and nothing
// verified their output. Runs meaningfully under scheduling-related env vars
// (e.g. YNN_SPLIT_TASKS) since those change task decomposition, not the
// graph.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/attention_graph.h"
#include "ynnpack/subgraph/test/scheduler.h"

namespace ynn {
namespace {

using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

// Naive reference: O[b,n,t,h] = softmax(scale * Q @ K^T) @ V with
// head-major Q/O [b,n,t,h] and K/V [b,n,s_buf,h]; only the first `s` of the
// s_buf cache entries participate (s < s_buf models a sliced KV cache).
void reference_attention(size_t b, size_t n, size_t t, size_t s, size_t s_buf,
                         size_t h, float scale, const std::vector<float>& q,
                         const std::vector<float>& k,
                         const std::vector<float>& v, std::vector<float>& o) {
  std::vector<double> logits(s);
  for (size_t bi = 0; bi < b; ++bi) {
    for (size_t ni = 0; ni < n; ++ni) {
      const float* kb = &k[((bi * n) + ni) * s_buf * h];
      const float* vb = &v[((bi * n) + ni) * s_buf * h];
      for (size_t ti = 0; ti < t; ++ti) {
        const float* qr = &q[(((bi * n) + ni) * t + ti) * h];
        double max_logit = -1e30;
        for (size_t si = 0; si < s; ++si) {
          double dot = 0;
          for (size_t hi = 0; hi < h; ++hi) {
            dot += static_cast<double>(qr[hi]) * kb[si * h + hi];
          }
          logits[si] = dot * scale;
          max_logit = std::max(max_logit, logits[si]);
        }
        double sum = 0;
        for (size_t si = 0; si < s; ++si) {
          logits[si] = std::exp(logits[si] - max_logit);
          sum += logits[si];
        }
        float* orow = &o[(((bi * n) + ni) * t + ti) * h];
        for (size_t hi = 0; hi < h; ++hi) {
          double acc = 0;
          for (size_t si = 0; si < s; ++si) {
            acc += logits[si] * vb[si * h + hi];
          }
          orow[hi] = static_cast<float>(acc / sum);
        }
      }
    }
  }
}

struct AttentionParams {
  size_t b, n, t, s, h;
  bool decode1;
  bool transpose_io;
  int threads;
  // If nonzero, K/V are sliced to this length with slice_like before the
  // attention (mirroring a KV cache whose buffer is s long but only s_active
  // entries are valid), and the graph is built with dynamic shapes.
  size_t s_active = 0;
};

void transpose_bnth_to_btnh(size_t b, size_t d1, size_t d2, size_t h,
                            const std::vector<float>& in,
                            std::vector<float>& out) {
  // [b, d1, d2, h] -> [b, d2, d1, h]
  for (size_t bi = 0; bi < b; ++bi)
    for (size_t i = 0; i < d1; ++i)
      for (size_t j = 0; j < d2; ++j)
        for (size_t hi = 0; hi < h; ++hi)
          out[(((bi * d2) + j) * d1 + i) * h + hi] =
              in[(((bi * d1) + i) * d2 + j) * h + hi];
}

void RunAttentionTest(const AttentionParams& p) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(p.h));
  const bool sliced = p.s_active != 0;
  const size_t s_used = sliced ? p.s_active : p.s;
  const bool dynamic = sliced;

  TestScheduler scheduler(p.threads - 1);
  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                        &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  subgraph_ptr subgraph = create_subgraph(sliced ? 5 : 4, 0);
  ASSERT_NE(subgraph, nullptr);

  const size_t qo_dims[] = {p.b, p.transpose_io ? p.t : p.n,
                            p.transpose_io ? p.n : p.t, p.h};
  const size_t kv_dims[] = {p.b, p.transpose_io ? p.s : p.n,
                            p.transpose_io ? p.n : p.s, p.h};
  const size_t kv_active_dims[] = {p.b, p.transpose_io ? s_used : p.n,
                                   p.transpose_io ? p.n : s_used, p.h};
  uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3, dummy_kv_id = 4;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                              dynamic ? nullptr : qo_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &q_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                              dynamic ? nullptr : kv_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &k_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                              dynamic ? nullptr : kv_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &v_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                              dynamic ? nullptr : qo_dims, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &o_id),
            ynn_status_success);

  uint32_t actual_k_id = k_id;
  uint32_t actual_v_id = v_id;
  // For decode (t == 1) the output is a single full row; only K/V are sliced.
  uint32_t actual_o_id = o_id;
  if (sliced) {
    ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                                dynamic ? nullptr : kv_active_dims, nullptr,
                                YNN_VALUE_FLAG_EXTERNAL_INPUT, &dummy_kv_id),
              ynn_status_success);
    int32_t seq_axis = p.transpose_io ? 1 : 2;
    uint32_t sliced_k_id = YNN_INVALID_VALUE_ID;
    uint32_t sliced_v_id = YNN_INVALID_VALUE_ID;
    ASSERT_EQ(ynn_define_slice_like(subgraph.get(), 1, &seq_axis, k_id,
                                    dummy_kv_id, &sliced_k_id, 0),
              ynn_status_success);
    ASSERT_EQ(ynn_define_slice_like(subgraph.get(), 1, &seq_axis, v_id,
                                    dummy_kv_id, &sliced_v_id, 0),
              ynn_status_success);
    actual_k_id = sliced_k_id;
    actual_v_id = sliced_v_id;
  }

  ynn_status status;
  if (p.decode1) {
    status = define_attention_decode1(subgraph.get(), q_id, actual_k_id,
                                      actual_v_id, scale, actual_o_id,
                                      p.transpose_io);
  } else {
    status = define_attention(subgraph.get(), q_id, actual_k_id, actual_v_id,
                              scale, actual_o_id, p.transpose_io);
  }
  ASSERT_EQ(status, ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0),
            ynn_status_success);
  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool.get(), 0);
  ASSERT_NE(runtime, nullptr);

  std::minstd_rand rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  // Head-major data for the reference.
  std::vector<float> q(p.b * p.n * p.t * p.h);
  std::vector<float> k(p.b * p.n * p.s * p.h);
  std::vector<float> v(p.b * p.n * p.s * p.h);
  for (auto& x : q) x = dist(rng);
  for (auto& x : k) x = dist(rng);
  for (auto& x : v) x = dist(rng);
  std::vector<float> o(p.b * p.n * p.t * p.h, 0.0f);

  // The graph's external layout depends on transpose_io.
  std::vector<float> q_ext = q, k_ext = k, v_ext = v;
  if (p.transpose_io) {
    transpose_bnth_to_btnh(p.b, p.n, p.t, p.h, q, q_ext);
    transpose_bnth_to_btnh(p.b, p.n, p.s, p.h, k, k_ext);
    transpose_bnth_to_btnh(p.b, p.n, p.s, p.h, v, v_ext);
  }

  if (dynamic) {
    ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), q_id, 4, qo_dims),
              ynn_status_success);
    ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), k_id, 4, kv_dims),
              ynn_status_success);
    ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), v_id, 4, kv_dims),
              ynn_status_success);
    if (sliced) {
      ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), dummy_kv_id, 4,
                                             kv_active_dims),
                ynn_status_success);
    }
  }
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), q_id, q_ext.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), k_id, k_ext.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), v_id, v_ext.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), o_id, o.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  std::vector<float> o_ref(o.size());
  reference_attention(p.b, p.n, p.t, s_used, p.s, p.h, scale, q, k, v, o_ref);
  std::vector<float> o_head = o;
  if (p.transpose_io) {
    // Output came back [b, t, n, h]; convert to head-major for comparison.
    transpose_bnth_to_btnh(p.b, p.t, p.n, p.h, o, o_head);
  }

  size_t mismatches = 0;
  for (size_t i = 0; i < o_ref.size(); ++i) {
    const float tolerance = 1e-4f + 1e-3f * std::abs(o_ref[i]);
    if (std::abs(o_head[i] - o_ref[i]) > tolerance) {
      if (++mismatches <= 5) {
        ADD_FAILURE() << "output[" << i << "] = " << o_head[i]
                      << ", expected " << o_ref[i];
      }
    }
  }
  EXPECT_EQ(mismatches, 0u);
}

struct TestName {
  template <typename T>
  std::string operator()(const T& info) const {
    const AttentionParams& p = info.param;
    return (p.decode1 ? std::string("decode1") : std::string("attention")) +
           "_s" + std::to_string(p.s) + "_t" + std::to_string(p.t) + "_n" +
           std::to_string(p.n) + "_h" + std::to_string(p.h) +
           (p.transpose_io ? "_tio" : "") + "_th" + std::to_string(p.threads) +
           (p.s_active ? "_sliced" + std::to_string(p.s_active) : "");
  }
};

class AttentionCorrectness : public testing::TestWithParam<AttentionParams> {};

TEST_P(AttentionCorrectness, MatchesReference) { RunAttentionTest(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    All, AttentionCorrectness,
    testing::ValuesIn<AttentionParams>({
        // b, n, t, s, h, decode1, transpose_io, threads, s_active
        {1, 4, 1, 33, 256, true, false, 4},
        {1, 4, 1, 527, 256, true, false, 4},
        {1, 4, 1, 4096, 256, true, false, 4},
        {1, 4, 1, 527, 256, true, true, 4},
        {1, 4, 1, 4096, 256, true, true, 4},
        {1, 8, 1, 1024, 64, true, true, 4},
        {1, 4, 1, 527, 256, true, true, 6},
        {1, 4, 1, 33, 256, false, false, 4},
        {1, 4, 1, 527, 256, false, true, 4},
        {1, 4, 128, 527, 256, false, true, 4},
        {1, 8, 64, 512, 64, false, false, 4},
        // Sliced dynamic KV cache (mirrors a decode step at partial context).
        {1, 4, 1, 4096, 256, true, false, 4, 527},
        {1, 4, 1, 4096, 256, true, true, 4, 527},
        {1, 4, 1, 4096, 256, true, true, 4, 33},
        {1, 4, 1, 4096, 256, true, true, 6, 1000},
        {1, 4, 1, 4096, 256, false, true, 4, 527},
    }),
    TestName());

}  // namespace
}  // namespace ynn
