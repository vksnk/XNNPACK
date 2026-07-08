// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

// Layout: Q, O are [b, n, t, h]; K, V are [b, n, s, h]. `block_width` == 0
// benchmarks the vanilla `define_attention` composite; otherwise
// `define_flash_attention`.
//
// `query_len` == 0 is the prefill / self-attention case (t == s == range(0)).
// A non-zero `query_len` fixes the query length and takes range(0) as the KV
// context length s: `query_len` == 1 is the autoregressive decoding case (a
// single query token attending over the whole KV cache).
// When `transpose_io` is set the external tensors are sequence-major
// (Q/O [b, t, n, h], K/V [b, s, n, h]) and `define_attention` inserts the
// transposes to head-major, mirroring XNNPACK's layout. Only valid for the
// vanilla path (block_width == 0).
void BenchAttention(benchmark::State& state, size_t b, size_t block_width,
                    size_t query_len = 0, bool transpose_io = false) {
  const size_t s = state.range(0);
  const size_t t = query_len == 0 ? s : query_len;
  const size_t h = state.range(1);
  const size_t n = state.range(2);
  const int num_threads = static_cast<int>(state.range(3));
  if (block_width != 0 && s % block_width != 0) {
    state.SkipWithError("s must be divisible by the block width");
    return;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  TestScheduler scheduler(num_threads);
  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                        &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  subgraph_ptr subgraph = create_subgraph(4, 0);
  if (!subgraph) {
    state.SkipWithError("failed to create subgraph");
    return;
  }

  // Define the external tensors with static shapes: the scheduler and kernel
  // selection make better decisions when the extents are known. Sequence-major
  // (b, seq, n, h) when transpose_io, head-major (b, n, seq, h) otherwise.
  const size_t qo_dims[] = {b, transpose_io ? t : n, transpose_io ? n : t, h};
  const size_t kv_dims[] = {b, transpose_io ? s : n, transpose_io ? n : s, h};
  uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &q_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &k_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &v_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &o_id);

  ynn_status status;
  if (block_width == 0) {
    status = define_attention(subgraph.get(), q_id, k_id, v_id, scale, o_id,
                              transpose_io);
  } else {
    status = define_flash_attention(subgraph.get(), q_id, k_id, v_id, scale,
                                    block_width, o_id);
  }
  if (status != ynn_status_success) {
    state.SkipWithError("failed to define attention");
    return;
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0) !=
      ynn_status_success) {
    state.SkipWithError("failed to optimize subgraph");
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph, threadpool.get(), 0);
  if (!runtime) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  std::vector<float> q(b * n * t * h, 0.01f);
  std::vector<float> k(b * n * s * h, 0.01f);
  std::vector<float> v(b * n * s * h, 0.01f);
  std::vector<float> o(b * n * t * h);

  if (ynn_set_external_value_shape(runtime.get(), q_id, 4, qo_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), k_id, 4, kv_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), v_id, 4, kv_dims) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), q_id, q.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), k_id, k.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), v_id, v.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(runtime.get(), o_id, o.data()) !=
          ynn_status_success) {
    state.SkipWithError("failed to set external values");
    return;
  }

  if (ynn_reshape_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  for (auto _ : state) {
    if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
      state.SkipWithError("failed to invoke runtime");
      return;
    }
  }

  const size_t flops = 2ull * b * n * t * s * h * 2;  // QK^T and P@V
  state.counters["FLOP"] = benchmark::Counter(
      static_cast<double>(state.iterations() * flops),
      benchmark::Counter::kIsRate);
}

void Attention(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/0);
}

// Same as Attention/AttentionDecode but with sequence-major I/O so the
// composite pays the boundary transposes XNNPACK's attention subgraph pays.
// A reference point for how much of the composite's edge is the head-major
// layout assumption vs. the streaming pipeline.
void AttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/0, /*query_len=*/0,
                 /*transpose_io=*/true);
}

void AttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/0, /*query_len=*/1,
                 /*transpose_io=*/true);
}

// The best block width depends on the L3 size: the softmax chain makes
// several passes over a seq x block_width score slab per block, so the slab
// should be a comfortably small fraction of L3.
void FlashAttention64(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/64);
}

void FlashAttention128(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/128);
}

void FlashAttention256(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/256);
}

void FlashAttention512(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/512);
}

// Decoding case: a single query token attends over a range(0)-long KV cache.
// The score slab is 1 x s, so vanilla never materializes a large scores matrix
// and the workload is dominated by streaming K and V once each (memory-bound).
void AttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/0, /*query_len=*/1);
}

void FlashAttentionDecode64(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/64, /*query_len=*/1);
}

void FlashAttentionDecode128(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/128, /*query_len=*/1);
}

void FlashAttentionDecode256(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/256, /*query_len=*/1);
}

void FlashAttentionDecode512(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/512, /*query_len=*/1);
}

void AttentionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"seq", "head", "heads", "threads"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  // {seq (=t=s), head_dim, num_heads} x thread sweep. Registering unused
  // configurations is free; use --benchmark_filter to select what to run.
  std::vector<std::vector<int>> shapes = {{256, 64, 8}, {512, 64, 8}, {1024, 64, 8}, {1024, 64, 32}, {4096, 64, 32}};
  for (const auto& shape : shapes) {
    for (int threads : {1, 2, 4}) {
      b->Args({shape[0], shape[1], shape[2], threads});
    }
  }
}

BENCHMARK(Attention)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention64)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention128)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention256)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention512)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionTransposed)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecodeTransposed)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecode)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode64)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode128)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode256)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode512)->Apply(AttentionArguments)->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynn
