// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Measures how long it takes to turn a subgraph into a runtime, which is what
// a delegate pays once for every partition it takes. A delegated model is made
// of many small partitions -- gemma-4-E2B-it through litert-lm builds 2140 of
// them, a median of 5 operations each, and spends 6.4 s doing it -- so the
// cost of building a *small* graph is what matters, not the cost per
// operation.

#include <atomic>
#include <cstddef>
#include <thread>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

constexpr size_t kTokens = 1024;
constexpr size_t kChannels = 640;

// A chain of elementwise operations, the shape of most of the partitions a
// delegate ends up with.
subgraph_ptr MakeElementwiseSubgraph(int num_ops) {
  ynn_subgraph_t raw = nullptr;
  ynn_create_subgraph(2, 0, &raw);
  subgraph_ptr subgraph(raw, &ynn_delete_subgraph);

  const size_t dims[] = {kTokens, kChannels};
  uint32_t in_id = 0;
  uint32_t out_id = 1;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);

  const float scalar = 1.5f;
  uint32_t scalar_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 0, nullptr, &scalar,
                    YNN_VALUE_FLAG_COPY_DATA, &scalar_id);

  uint32_t current_id = in_id;
  for (int i = 0; i < num_ops; ++i) {
    const bool last = i == num_ops - 1;
    uint32_t next_id = last ? out_id : YNN_INVALID_VALUE_ID;
    if (i % 3 == 2) {
      ynn_define_unary(subgraph.get(), ynn_unary_rsqrt, current_id, &next_id, 0);
    } else {
      ynn_define_binary(subgraph.get(),
                        i % 3 == 0 ? ynn_binary_multiply : ynn_binary_add,
                        current_id, scalar_id, &next_id, 0);
    }
    current_id = next_id;
  }
  return subgraph;
}

// A quantized matmul with the normalization around it, i.e. a partition that
// carries real work rather than only elementwise operations.
subgraph_ptr MakeDotSubgraph() {
  ynn_subgraph_t raw = nullptr;
  ynn_create_subgraph(2, 0, &raw);
  subgraph_ptr subgraph(raw, &ynn_delete_subgraph);

  static std::vector<int8_t> weights(kChannels * kChannels, 1);

  const size_t dims[] = {kTokens, kChannels};
  const size_t weight_dims[] = {kChannels, kChannels};
  uint32_t in_id = 0;
  uint32_t out_id = 1;
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id);
  ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);

  uint32_t weights_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, weight_dims,
                    weights.data(), 0, &weights_id);

  const int32_t axis = -1;
  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  ynn_define_reduce(subgraph.get(), ynn_reduce_min_max, 1, &axis, in_id,
                    YNN_INVALID_VALUE_ID, &min_max_id,
                    YNN_NODE_FLAG_KEEP_DIMS);
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  ynn_define_dynamic_quantization(subgraph.get(), min_max_id, ynn_type_int8,
                                  &zero_point_id, &scale_id, 0);
  uint32_t quantized_id = YNN_INVALID_VALUE_ID;
  ynn_define_quantize(subgraph.get(), in_id, ynn_type_int8, zero_point_id,
                      scale_id, &quantized_id, 0);
  uint32_t dot_id = YNN_INVALID_VALUE_ID;
  ynn_define_dot(subgraph.get(), 1, quantized_id, weights_id,
                 YNN_INVALID_VALUE_ID, &dot_id, 0);
  uint32_t out = out_id;
  ynn_define_dequantize(subgraph.get(), dot_id, zero_point_id, scale_id,
                        ynn_type_fp32, &out, 0);
  return subgraph;
}

void RunBuildBenchmark(benchmark::State& state, subgraph_ptr subgraph,
                       int thread_count) {
  TestScheduler scheduler(thread_count - 1);
  ynn_threadpool_t threadpool_raw = nullptr;
  ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                        &threadpool_raw);
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  if (ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0) !=
      ynn_status_success) {
    state.SkipWithError("failed to optimize subgraph");
    return;
  }

  for (auto _ : state) {
    ynn_runtime_t runtime = nullptr;
    if (ynn_create_runtime(subgraph.get(), threadpool.get(), 0, &runtime) !=
        ynn_status_success) {
      state.SkipWithError("failed to create runtime");
      return;
    }
    ynn_delete_runtime(runtime);
  }
}

// The builds of different partitions are independent of each other, and a
// delegate has many of them to do. This measures whether doing them at the
// same time works and how well it scales.
void BM_CreateRuntimeConcurrent(benchmark::State& state) {
  const int num_graphs = state.range(0);
  const int num_threads = state.range(1);

  std::vector<subgraph_ptr> subgraphs;
  std::vector<threadpool_ptr> threadpools;
  std::vector<std::unique_ptr<TestScheduler>> schedulers;
  for (int i = 0; i < num_graphs; ++i) {
    subgraphs.push_back(MakeElementwiseSubgraph(5));
    schedulers.push_back(std::make_unique<TestScheduler>(0));
    ynn_threadpool_t raw = nullptr;
    ynn_create_threadpool(TestScheduler::scheduler(), schedulers.back().get(),
                          0, &raw);
    threadpools.emplace_back(raw, &ynn_delete_threadpool);
    if (ynn_optimize_subgraph(subgraphs.back().get(),
                              threadpools.back().get(),
                              0) != ynn_status_success) {
      state.SkipWithError("failed to optimize subgraph");
      return;
    }
  }

  for (auto _ : state) {
    std::atomic<int> next{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        for (int i = next++; i < num_graphs; i = next++) {
          ynn_runtime_t runtime = nullptr;
          if (ynn_create_runtime(subgraphs[i].get(), threadpools[i].get(), 0,
                                 &runtime) != ynn_status_success) {
            ++failures;
            continue;
          }
          ynn_delete_runtime(runtime);
        }
      });
    }
    for (std::thread& thread : threads) thread.join();
    if (failures > 0) {
      state.SkipWithError("failed to create runtime");
      return;
    }
  }
}

void BM_CreateRuntimeElementwise(benchmark::State& state) {
  RunBuildBenchmark(state, MakeElementwiseSubgraph(state.range(0)),
                    state.range(1));
}

void BM_CreateRuntimeDot(benchmark::State& state) {
  RunBuildBenchmark(state, MakeDotSubgraph(), state.range(0));
}

BENCHMARK(BM_CreateRuntimeElementwise)
    ->ArgNames({"ops", "threads"})
    ->Args({1, 1})
    ->Args({5, 1})
    ->Args({20, 1})
    ->Args({50, 1})
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_CreateRuntimeConcurrent)
    ->ArgNames({"graphs", "threads"})
    ->Args({64, 1})
    ->Args({64, 2})
    ->Args({64, 4})
    ->Args({64, 6})
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(BM_CreateRuntimeDot)
    ->ArgNames({"threads"})
    ->Args({1})
    ->Unit(benchmark::TimeUnit::kMicrosecond);

}  // namespace
}  // namespace ynn
