// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Memory-efficient ("flash") attention expressed declaratively with the YNNPACK
// subgraph API, using a two-pass packed rfactor.
//
// Pass 1 (fused map): chop the key/value sequence N into `K_tiles` blocks of
// width `w`, and for each block independently compute the block-local max
// (m_local), the block-local softmax denominator (l_local), and the
// un-normalized block output (O_local = P_local @ V). These are *packed* into a
// single tensor of shape [..., K_tiles, t, h + 2] by concatenating m_local,
// l_local and O_local along the last (feature) axis. The packing introduces an
// artificial data dependency between the three quantities so they share a
// single consumer; the goal is to force Slinky's LCA bounds inference to fuse
// Pass 1 into one inner block loop and to bound the intermediate allocation.
//
// Pass 2 (stateless spatial reduction): slice the packed tensor back into
// views, reduce the global max across blocks, rescale each block by
// exp(m_local - m_global), and spatially accumulate the denominator and
// numerator before the final normalization.
//
// This file contains both a correctness test (gated against a plain C++
// reference) and a benchmark; see main() at the bottom.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

// Layout: Q,O are [b, n, t, h]; K,V are [b, n, s, h].
struct AttentionDims {
  size_t b;  // batch
  size_t n;  // heads
  size_t t;  // query sequence length
  size_t h;  // head dim
  size_t s;  // key/value sequence length
  size_t w;  // block width (s must be divisible by w)

  size_t k_tiles() const { return s / w; }
};

// Plain, numerically-stable scaled dot-product attention reference.
Tensor<float> ReferenceAttention(const Tensor<float>& q, const Tensor<float>& k,
                                 const Tensor<float>& v, float scale) {
  const size_t b = q.extent(0), n = q.extent(1), t = q.extent(2),
               h = q.extent(3);
  const size_t s = k.extent(2);
  Tensor<float> o({b, n, t, h});
  std::vector<float> scores(s);
  for (size_t bi = 0; bi < b; ++bi) {
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t i = 0; i < t; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < s; ++j) {
          float dot = 0.0f;
          for (size_t d = 0; d < h; ++d) {
            dot += q(bi, ni, i, d) * k(bi, ni, j, d);
          }
          scores[j] = dot * scale;
          m = std::max(m, scores[j]);
        }
        float l = 0.0f;
        for (size_t j = 0; j < s; ++j) {
          scores[j] = std::exp(scores[j] - m);
          l += scores[j];
        }
        for (size_t d = 0; d < h; ++d) {
          float acc = 0.0f;
          for (size_t j = 0; j < s; ++j) {
            acc += scores[j] * v(bi, ni, j, d);
          }
          o(bi, ni, i, d) = acc / l;
        }
      }
    }
  }
  return o;
}

// Builds the packed two-pass flash attention graph. Q/K/V are external inputs
// with ids 0/1/2, O is external output id 3.
void BuildPackedFlashAttention(SubgraphBuilder& sg, const AttentionDims& d,
                               float scale) {
  const size_t b = d.b, n = d.n, t = d.t, h = d.h, s = d.s, w = d.w;
  const size_t B = d.k_tiles();

  const uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3;
  sg.AddInput(ynn_type_fp32, {b, n, t, h}, q_id)
      .AddInput(ynn_type_fp32, {b, n, s, h}, k_id)
      .AddInput(ynn_type_fp32, {b, n, s, h}, v_id)
      .AddOutput(ynn_type_fp32, {b, n, t, h}, o_id);

  const uint32_t scale_id = sg.DefineScalar(scale);

  // Intermediate tensor ids.
  uint32_t qs = YNN_INVALID_VALUE_ID, qs5 = YNN_INVALID_VALUE_ID,
           kt = YNN_INVALID_VALUE_ID, ktt = YNN_INVALID_VALUE_ID,
           vt = YNN_INVALID_VALUE_ID, sc = YNN_INVALID_VALUE_ID,
           m_local = YNN_INVALID_VALUE_ID, sm = YNN_INVALID_VALUE_ID,
           p_local = YNN_INVALID_VALUE_ID, l_local = YNN_INVALID_VALUE_ID,
           o_unnorm = YNN_INVALID_VALUE_ID, packed = YNN_INVALID_VALUE_ID,
           loc_max = YNN_INVALID_VALUE_ID, loc_sum = YNN_INVALID_VALUE_ID,
           loc_out = YNN_INVALID_VALUE_ID, m_global = YNN_INVALID_VALUE_ID,
           diff = YNN_INVALID_VALUE_ID, scale_k = YNN_INVALID_VALUE_ID,
           sum_scaled = YNN_INVALID_VALUE_ID, l_global = YNN_INVALID_VALUE_ID,
           out_scaled = YNN_INVALID_VALUE_ID, num = YNN_INVALID_VALUE_ID;

  sg.AddTensor(ynn_type_fp32, {b, n, t, h}, qs)
      .AddTensor(ynn_type_fp32, {b, n, 1, t, h}, qs5)
      .AddTensor(ynn_type_fp32, {b, n, B, w, h}, kt)
      .AddTensor(ynn_type_fp32, {b, n, B, h, w}, ktt)
      .AddTensor(ynn_type_fp32, {b, n, B, w, h}, vt)
      .AddTensor(ynn_type_fp32, {b, n, B, t, w}, sc)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, m_local)
      .AddTensor(ynn_type_fp32, {b, n, B, t, w}, sm)
      .AddTensor(ynn_type_fp32, {b, n, B, t, w}, p_local)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, l_local)
      .AddTensor(ynn_type_fp32, {b, n, B, t, h}, o_unnorm)
      .AddTensor(ynn_type_fp32, {b, n, B, t, h + 2}, packed)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, loc_max)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, loc_sum)
      .AddTensor(ynn_type_fp32, {b, n, B, t, h}, loc_out)
      .AddTensor(ynn_type_fp32, {b, n, 1, t, 1}, m_global)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, diff)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, scale_k)
      .AddTensor(ynn_type_fp32, {b, n, B, t, 1}, sum_scaled)
      .AddTensor(ynn_type_fp32, {b, n, t, 1}, l_global)
      .AddTensor(ynn_type_fp32, {b, n, B, t, h}, out_scaled)
      .AddTensor(ynn_type_fp32, {b, n, t, h}, num);

  // ---- Pass 1: fused per-block map & pack ----
  sg.AddBinary(ynn_binary_multiply, q_id, scale_id, qs);
  sg.AddReshape({b, n, 1, t, h}, qs, qs5);
  sg.AddReshape({b, n, B, w, h}, k_id, kt);
  sg.AddTranspose({0, 1, 2, 4, 3}, kt, ktt);  // [.., w, h] -> [.., h, w]
  sg.AddReshape({b, n, B, w, h}, v_id, vt);

  // S_local = (Q @ K^T) * scale, broadcasting the block axis of Q over B.
  sg.AddDot(1, qs5, ktt, YNN_INVALID_VALUE_ID, sc);
  sg.AddReduce(ynn_reduce_max, {4}, sc, YNN_INVALID_VALUE_ID, m_local,
               YNN_NODE_FLAG_KEEP_DIMS);
  sg.AddBinary(ynn_binary_subtract, sc, m_local, sm);
  sg.AddUnary(ynn_unary_exp, sm, p_local);
  sg.AddReduce(ynn_reduce_sum, {4}, p_local, YNN_INVALID_VALUE_ID, l_local,
               YNN_NODE_FLAG_KEEP_DIMS);
  sg.AddDot(1, p_local, vt, YNN_INVALID_VALUE_ID, o_unnorm);

  // Pack [m_local | l_local | O_local] along the feature axis.
  sg.AddConcatenate(4, {m_local, l_local, o_unnorm}, packed);

  // ---- Pass 2: slice & stateless spatial reduction ----
  sg.AddSlice({4}, {0}, {1}, {1}, packed, loc_max);
  sg.AddSlice({4}, {1}, {2}, {1}, packed, loc_sum);
  sg.AddSlice({4}, {2}, {static_cast<int64_t>(h + 2)}, {1}, packed, loc_out);

  sg.AddReduce(ynn_reduce_max, {2}, loc_max, YNN_INVALID_VALUE_ID, m_global,
               YNN_NODE_FLAG_KEEP_DIMS);
  sg.AddBinary(ynn_binary_subtract, loc_max, m_global, diff);
  sg.AddUnary(ynn_unary_exp, diff, scale_k);

  sg.AddBinary(ynn_binary_multiply, loc_sum, scale_k, sum_scaled);
  // The last two reductions drop the (kept elsewhere) block axis entirely, so
  // the final divide produces the external output directly with no trailing
  // reshape/copy.
  sg.AddReduce(ynn_reduce_sum, {2}, sum_scaled, YNN_INVALID_VALUE_ID, l_global);

  sg.AddBinary(ynn_binary_multiply, loc_out, scale_k, out_scaled);
  sg.AddReduce(ynn_reduce_sum, {2}, out_scaled, YNN_INVALID_VALUE_ID, num);

  sg.AddBinary(ynn_binary_divide, num, l_global, o_id);
}

// ---- Correctness test ----

void RunCorrectness(const AttentionDims& d) {
  ReplicableRandomDevice rng;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d.h));

  SubgraphBuilder sg(4);
  BuildPackedFlashAttention(sg, d, scale);

  Runtime runtime(sg.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> q({d.b, d.n, d.t, d.h});
  Tensor<float> k({d.b, d.n, d.s, d.h});
  Tensor<float> v({d.b, d.n, d.s, d.h});
  fill_random(q.data(), q.size(), rng, -1.0f, 1.0f);
  fill_random(k.data(), k.size(), rng, -1.0f, 1.0f);
  fill_random(v.data(), v.size(), rng, -1.0f, 1.0f);

  Tensor<float> o({d.b, d.n, d.t, d.h});
  runtime.ReshapeExternalTensor({d.b, d.n, d.t, d.h}, q.data(), 0)
      .ReshapeExternalTensor({d.b, d.n, d.s, d.h}, k.data(), 1)
      .ReshapeExternalTensor({d.b, d.n, d.s, d.h}, v.data(), 2);
  runtime.SetupExternalTensor(o.data(), 3);
  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> expected = ReferenceAttention(q, k, v, scale);
  for (const auto& i : EnumerateIndices(o.extents())) {
    ASSERT_NEAR(o(i), expected(i), 1e-3f) << "at " << index_to_string(i);
  }
}

TEST(Attention, PackedRfactorMatchesReference) {
  RunCorrectness({/*b=*/2, /*n=*/2, /*t=*/8, /*h=*/16, /*s=*/32, /*w=*/8});
  RunCorrectness({/*b=*/1, /*n=*/3, /*t=*/5, /*h=*/16, /*s=*/64, /*w=*/16});
  RunCorrectness({/*b=*/2, /*n=*/1, /*t=*/7, /*h=*/32, /*s=*/128, /*w=*/64});
  // Single block (B == 1): degenerates to ordinary attention.
  RunCorrectness({/*b=*/1, /*n=*/1, /*t=*/4, /*h=*/16, /*s=*/16, /*w=*/16});
}

// ---- Benchmark ----

void BenchPackedFlashAttention(benchmark::State& state) {
  const AttentionDims d{/*b=*/1,
                        /*n=*/static_cast<size_t>(state.range(2)),
                        /*t=*/static_cast<size_t>(state.range(0)),
                        /*h=*/static_cast<size_t>(state.range(1)),
                        /*s=*/static_cast<size_t>(state.range(0)),
                        /*w=*/static_cast<size_t>(state.range(3))};
  if (d.s % d.w != 0) {
    state.SkipWithError("s must be divisible by w");
    return;
  }
  const int num_threads = static_cast<int>(state.range(4));
  const float scale = 1.0f / std::sqrt(static_cast<float>(d.h));

  SubgraphBuilder sg(4);
  BuildPackedFlashAttention(sg, d, scale);
  TestScheduler scheduler(num_threads);
  Runtime runtime(sg.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  Tensor<float> q({d.b, d.n, d.t, d.h});
  Tensor<float> k({d.b, d.n, d.s, d.h});
  Tensor<float> v({d.b, d.n, d.s, d.h});
  Tensor<float> o({d.b, d.n, d.t, d.h});
  q.fill(0.01f);
  k.fill(0.01f);
  v.fill(0.01f);
  runtime.ReshapeExternalTensor({d.b, d.n, d.t, d.h}, q.data(), 0)
      .ReshapeExternalTensor({d.b, d.n, d.s, d.h}, k.data(), 1)
      .ReshapeExternalTensor({d.b, d.n, d.s, d.h}, v.data(), 2);
  runtime.SetupExternalTensor(o.data(), 3);
  runtime.ReshapeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  for (auto _ : state) {
    runtime.InvokeRuntime();
  }

  const size_t flops =
      2ull * d.b * d.n * d.t * d.s * d.h * 2;  // QK^T and P@V
  state.counters["FLOP"] = benchmark::Counter(
      static_cast<double>(state.iterations() * flops),
      benchmark::Counter::kIsRate);
}

}  // namespace
}  // namespace ynn

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  benchmark::Initialize(&argc, argv);

  const int test_result = RUN_ALL_TESTS();
  if (test_result != 0) {
    return test_result;
  }

  auto* bench = benchmark::RegisterBenchmark(
      "PackedFlashAttention", ynn::BenchPackedFlashAttention);
  bench->ArgNames({"seq", "head", "heads", "w", "threads"});
  bench->UseRealTime();
  bench->MeasureProcessCPUTime();
  // {seq (=t=s), head_dim, num_heads, block_width} x thread sweep
  for (int threads : {1, 4, 8, 16}) {
    bench->Args({1024, 64, 32, 256, threads});
    bench->Args({4096, 64, 32, 256, threads});
    bench->Args({4096, 64, 32, 512, threads});
    bench->Args({4096, 64, 32, 1024, threads});
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
