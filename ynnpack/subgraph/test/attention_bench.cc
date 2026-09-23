// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/test/attention_graph.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

// Tracks the heap memory held by the runtime's scratch buffers.
std::atomic<size_t> g_current_bytes{0};
std::atomic<size_t> g_peak_bytes{0};
std::mutex g_sizes_mutex;
std::unordered_map<void*, size_t> g_sizes;

void* TrackingAlloc(size_t alignment, size_t size) {
  void* ptr = std::aligned_alloc(alignment, (size + alignment - 1) /
                                                alignment * alignment);
  if (!ptr) return nullptr;
  {
    std::lock_guard<std::mutex> lock(g_sizes_mutex);
    g_sizes[ptr] = size;
  }
  const size_t current = g_current_bytes += size;
  size_t peak = g_peak_bytes.load();
  while (current > peak && !g_peak_bytes.compare_exchange_weak(peak, current)) {
  }
  return ptr;
}

void TrackingFree(void* ptr) {
  if (!ptr) return;
  size_t size;
  {
    std::lock_guard<std::mutex> lock(g_sizes_mutex);
    auto it = g_sizes.find(ptr);
    size = it->second;
    g_sizes.erase(it);
  }
  g_current_bytes -= size;
  std::free(ptr);
}

using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

// Layout: Q, O are [b, n, t, h]; K, V are [b, n, s, h].
//
// `query_len` == 0 is the prefill / self-attention case (t == s == range(0)).
// A non-zero `query_len` fixes the query length and takes range(0) as the KV
// context length s: `query_len` == 1 is the autoregressive decoding case (a
// single query token attending over the whole KV cache).
// When `transpose_io` is set the external tensors are sequence-major
// (Q/O [b, t, n, h], K/V [b, s, n, h]) and `define_attention` inserts the
// transposes to head-major, mirroring XNNPACK's layout.
// `block_width` == 0 benchmarks the vanilla `define_attention` composite;
// otherwise `define_flash_attention` (head-major I/O and the full KV sequence
// only).
void BenchAttentionImpl(benchmark::State& state, size_t b, size_t query_len,
                        bool transpose_io, bool decode1, size_t block_width,
                        bool dynamic, size_t s, size_t h, size_t n,
                        int num_threads, int s_active,
                        bool flash_transposed = false) {
  const size_t t = query_len == 0 ? s : query_len;
  s_active = std::min<int>(s, s_active);
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));
  if (block_width != 0) {
    if (transpose_io || decode1 || s_active != s) {
      state.SkipWithError("unsupported configuration for flash attention");
      return;
    }
    if (s % block_width != 0) {
      state.SkipWithError("s must be divisible by the block width");
      return;
    }
  }

  // The `threads` argument is the total number of threads that should run the
  // work. The runtime's invoking thread participates as a worker, so the
  // scheduler only needs `num_threads - 1` background threads.
  TestScheduler scheduler(num_threads - 1);

  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                        &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  subgraph_ptr subgraph = create_subgraph(s_active != s ? 5 : 4, 0);
  if (!subgraph) {
    state.SkipWithError("failed to create subgraph");
    return;
  }

  // Define the external tensors.
  const size_t qo_dims[] = {b, transpose_io ? t : n, transpose_io ? n : t, h};
  const size_t kv_dims[] = {b, transpose_io ? s : n, transpose_io ? n : s, h};
  const size_t kv_active_dims[] = {b, transpose_io ? s_active : n,
                                   transpose_io ? n : s_active, h};
  uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3, dummy_kv_id = 4;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &q_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &k_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : kv_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &v_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                    dynamic ? nullptr : qo_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &o_id);

  uint32_t actual_k_id = k_id;
  uint32_t actual_v_id = v_id;
  uint32_t actual_o_id = (s_active != s) ? YNN_INVALID_VALUE_ID : o_id;

  if (s_active != s) {
    // Slice the K and V values to the active sequence length in the reduction
    // dimension.
    ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4,
                      dynamic ? nullptr : kv_active_dims, nullptr,
                      YNN_VALUE_FLAG_EXTERNAL_INPUT, &dummy_kv_id);

    int32_t seq_axis = transpose_io ? 1 : 2;
    uint32_t sliced_k_id = YNN_INVALID_VALUE_ID;
    uint32_t sliced_v_id = YNN_INVALID_VALUE_ID;

    if (ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis, k_id,
                              dummy_kv_id, &sliced_k_id,
                              /*flags=*/0) != ynn_status_success ||
        ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis, v_id,
                              dummy_kv_id, &sliced_v_id,
                              /*flags=*/0) != ynn_status_success) {
      state.SkipWithError("failed to define slice_like for KV cache");
      return;
    }
    actual_k_id = sliced_k_id;
    actual_v_id = sliced_v_id;
  }

  ynn_status status;
  if (block_width != 0 && flash_transposed) {
    status = define_flash_attention_transposed(subgraph.get(), q_id,
                                               actual_k_id, actual_v_id, scale,
                                               block_width, actual_o_id);
  } else if (block_width != 0) {
    status = define_flash_attention(subgraph.get(), q_id, actual_k_id,
                                    actual_v_id, scale, block_width,
                                    actual_o_id);
  } else if (decode1) {
    status =
        define_attention_decode1(subgraph.get(), q_id, actual_k_id, actual_v_id,
                                 scale, actual_o_id, transpose_io);
  } else {
    status = define_attention(subgraph.get(), q_id, actual_k_id, actual_v_id,
                              scale, actual_o_id, transpose_io);
  }

  if (status != ynn_status_success) {
    state.SkipWithError("failed to define attention");
    return;
  }

  if (s_active != s) {
    // Slice the output to the active sequence length. Note that this keeps the
    // original shape, it just limits the range of the computation to the active
    // sequence length.
    int32_t seq_axis = transpose_io ? 1 : 2;
    if (ynn_define_slice_like(subgraph.get(), /*num_axes=*/1, &seq_axis,
                              actual_o_id, dummy_kv_id, &o_id,
                              YNN_NODE_FLAG_KEEP_SHAPE) != ynn_status_success) {
      state.SkipWithError("failed to define slice_like for attention output");
      return;
    }
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0) !=
      ynn_status_success) {
    state.SkipWithError("failed to optimize subgraph");
    return;
  }

  // Route the runtime's scratch allocations through the tracking allocator.
  // The hook is read when the runtime is created.
  ynn_set_buffer_allocator(TrackingAlloc, TrackingFree);
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
      (s_active != s &&
       ynn_set_external_value_shape(runtime.get(), dummy_kv_id, 4,
                                    kv_active_dims) != ynn_status_success) ||
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

  g_peak_bytes = g_current_bytes.load();
  for (auto _ : state) {
    if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
      state.SkipWithError("failed to invoke runtime");
      return;
    }
  }

  // Peak bytes of heap scratch memory held by the runtime during an invoke
  // (allocations below slinky's stack threshold are not included).
  state.counters["peak_MB"] =
      static_cast<double>(g_peak_bytes.load()) /
      (1024.0 * 1024.0);

  const size_t flops = 2ull * b * n * t * s_active * h * 2;  // QK^T and P@V
  state.counters["FLOP"] =
      benchmark::Counter(static_cast<double>(state.iterations() * flops),
                         benchmark::Counter::kIsRate);
}

void BenchAttention(benchmark::State& state, size_t b, size_t query_len = 0,
                    bool transpose_io = false, bool decode1 = false,
                    size_t block_width = 0) {
  BenchAttentionImpl(state, b, query_len, transpose_io, decode1, block_width,
                     /*dynamic=*/state.range(0), /*s=*/state.range(1),
                     /*h=*/state.range(2), /*n=*/state.range(3),
                     /*num_threads=*/static_cast<int>(state.range(4)),
                     /*s_active=*/static_cast<int>(state.range(5)));
}

void Attention(benchmark::State& state) { BenchAttention(state, /*b=*/1); }

// Prefill over a range of sequence lengths, with the flash block width as an
// argument (0 = vanilla attention).
void AttentionSweep(benchmark::State& state) {
  const size_t s = state.range(0);
  BenchAttentionImpl(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                     /*decode1=*/false, /*block_width=*/state.range(3),
                     /*dynamic=*/false, s, /*h=*/64, /*n=*/state.range(1),
                     /*num_threads=*/static_cast<int>(state.range(2)),
                     /*s_active=*/static_cast<int>(s),
                     /*flash_transposed=*/state.range(4));
}

void AttentionSweepArguments(benchmark::Benchmark* b) {
  b->ArgNames({"seq", "heads", "threads", "block", "transposed"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int seq : {1024, 2048, 4096, 8192, 16384}) {
    for (int heads : {8, 32}) {
      for (int threads : {1, 4}) {
        for (int block : {0, 64, 128, 256, 512, 1024, 2048}) {
          if (block >= seq) continue;
          b->Args({seq, heads, threads, block, 0});
          if (block != 0) b->Args({seq, heads, threads, block, 1});
        }
      }
    }
  }
}

// The best block width depends on the L3 size: the softmax chain makes
// several passes over a seq x block_width score slab per block, so the slab
// should be a comfortably small fraction of L3.
void FlashAttention64(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/64);
}

void FlashAttention128(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/128);
}

void FlashAttention256(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/256);
}

void FlashAttention512(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/512);
}

void AttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0,
                 /*transpose_io=*/true);
}

void AttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/true);
}

// Decoding case: a single query token attends over a range(0)-long KV cache.
// The score slab is 1 x s, so vanilla never materializes a large scores matrix
// and the workload is dominated by streaming K and V once each (memory-bound).
void AttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1);
}

// Same as AttentionDecode, but keeps K as the dot's `A` operand (natural
// layout, no transpose/pack) and makes Q the `B` operand instead (a free
// size-1-axis swap), avoiding the O(s * h) K-transpose/pack that
// AttentionDecode pays on every decode step. See define_attention_decode1.
void AttentionDecode1Transposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/true, /*decode1=*/true);
}

void AttentionDecode1(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/false, /*decode1=*/true);
}

void FlashAttentionDecode64(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/64);
}

void FlashAttentionDecode128(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/128);
}

void FlashAttentionDecode256(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/256);
}

void FlashAttentionDecode512(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 /*decode1=*/false, /*block_width=*/512);
}

void AttentionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"dynamic", "seq", "head", "heads", "threads", "seq_active"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  std::vector<std::vector<int>> shapes = {{256, 64, 8},
                                          {512, 64, 8},
                                          {1024, 64, 8},
                                          {1024, 64, 32},
                                          {4096, 64, 32}};
  for (bool dynamic : {false, true}) {
    for (const auto& shape : shapes) {
      for (int threads : {1, 2, 4}) {
        for (float s_fraction : {0.01f, 0.5f, 0.99f, 1.0f}) {
          const int s_active = std::ceil(s_fraction * shape[0]);
          b->Args({dynamic, shape[0], shape[1], shape[2], threads, s_active});
        }
      }
    }
  }
}

BENCHMARK(AttentionSweep)
    ->Apply(AttentionSweepArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(Attention)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention64)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention128)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention256)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttention512)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecodeTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecode)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecode1Transposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecode1)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode64)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode128)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode256)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FlashAttentionDecode512)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynn
