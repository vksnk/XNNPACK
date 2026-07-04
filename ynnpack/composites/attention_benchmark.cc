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
void BenchAttention(benchmark::State& state, size_t b, size_t block_width) {
  const size_t t = state.range(0);
  const size_t h = state.range(1);
  const size_t n = state.range(2);
  const size_t s = t;
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
  // selection make better decisions when the extents are known.
  const size_t qo_dims[] = {b, n, t, h};
  const size_t kv_dims[] = {b, n, s, h};
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
    status = define_attention(subgraph.get(), q_id, k_id, v_id, scale, o_id);
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

  const size_t qo_shape[] = {b, n, t, h};
  const size_t kv_shape[] = {b, n, s, h};
  if (ynn_set_external_value_shape(runtime.get(), q_id, 4, qo_shape) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), k_id, 4, kv_shape) !=
          ynn_status_success ||
      ynn_set_external_value_shape(runtime.get(), v_id, 4, kv_shape) !=
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

void FlashAttention256(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/256);
}

void FlashAttention512(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*block_width=*/512);
}

void AttentionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"seq", "head", "heads", "threads"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  // {seq (=t=s), head_dim, num_heads} x thread sweep
  for (int threads : {1, 8}) {
    b->Args({1024, 64, 32, threads});
    b->Args({4096, 64, 32, threads});
  }
}

BENCHMARK(Attention)->Apply(AttentionArguments);
BENCHMARK(FlashAttention256)->Apply(AttentionArguments);
BENCHMARK(FlashAttention512)->Apply(AttentionArguments);

}  // namespace
}  // namespace ynn
