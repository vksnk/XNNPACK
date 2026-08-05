// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

subgraph_ptr create_subgraph(size_t num_external_values, uint32_t flags = 0) {
  ynn_subgraph_t subgraph = nullptr;
  ynn_create_subgraph(num_external_values, flags, &subgraph);
  return subgraph_ptr(subgraph, &ynn_delete_subgraph);
}

runtime_ptr create_runtime(ynn_subgraph_t subgraph,
                           ynn_threadpool_t threadpool = nullptr,
                           uint32_t flags = 0) {
  ynn_runtime_t runtime = nullptr;
  ynn_create_runtime(subgraph, threadpool, flags, &runtime);
  return runtime_ptr(runtime, &ynn_delete_runtime);
}

threadpool_ptr create_threadpool(ynn_scheduler_t scheduler,
                                 void* scheduler_context, uint32_t flags = 0) {
  ynn_threadpool_t threadpool = nullptr;
  ynn_create_threadpool(scheduler, scheduler_context, flags, &threadpool);
  return threadpool_ptr(threadpool, &ynn_delete_threadpool);
}

// Benchmarks a softmax-style fused chain over the last axis of a 3D tensor:
// exp(x) / sum(exp(x), axis=2). With `plain` set, benchmarks a single
// standalone divide of the same shape instead, as an unfused control.
template <typename T>
void bench(benchmark::State& state, ynn_threadpool_t threadpool, int dim0,
           int dim1, int dim2, bool plain) {
  subgraph_ptr subgraph = create_subgraph(2);

  // If we have static shapes, set them now.
  size_t input_shape[3] = {0, 0, 0};
  if (dim0 > 0) input_shape[0] = dim0;
  if (dim1 > 0) input_shape[1] = dim1;
  if (dim2 > 0) input_shape[2] = dim2;

  uint32_t input_id = 0;
  uint32_t output_id = 1;
  if (ynn_define_tensor(subgraph.get(), type_of<T>(), 3, &input_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &input_id) != ynn_status_success) {
    state.SkipWithError("Failed to define input tensor");
    return;
  }
  if (ynn_define_tensor(subgraph.get(), type_of<T>(), 3, &input_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                        &output_id) != ynn_status_success) {
    state.SkipWithError("Failed to define output tensor");
    return;
  }

  if (plain) {
    // Standalone single-op control: output = input / input.
    if (ynn_define_binary(subgraph.get(), ynn_binary_divide, input_id, input_id,
                          &output_id, /*flags=*/0) != ynn_status_success) {
      state.SkipWithError("Failed to define divide node");
      return;
    }
  } else {
    uint32_t exp_id = YNN_INVALID_VALUE_ID;
    if (ynn_define_unary(subgraph.get(), ynn_unary_exp, input_id, &exp_id,
                         /*flags=*/0) != ynn_status_success) {
      state.SkipWithError("Failed to define exp node");
      return;
    }
    uint32_t sum_id = YNN_INVALID_VALUE_ID;
    int32_t axis = 2;
    if (ynn_define_reduce(subgraph.get(), ynn_reduce_sum, 1, &axis, exp_id,
                          YNN_INVALID_VALUE_ID, &sum_id,
                          /*flags=*/YNN_NODE_FLAG_KEEP_DIMS) !=
        ynn_status_success) {
      state.SkipWithError("Failed to define reduce node");
      return;
    }
    if (ynn_define_binary(subgraph.get(), ynn_binary_divide, exp_id, sum_id,
                          &output_id, /*flags=*/0) != ynn_status_success) {
      state.SkipWithError("Failed to define divide node");
      return;
    }
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0) !=
      ynn_status_success) {
    state.SkipWithError("Failed to optimize subgraph");
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);
  if (!runtime) {
    state.SkipWithError("Failed to create ynnpack runtime");
    return;
  }

  // A negative shape indicates a dynamic shape of the same magnitude.
  dim0 = std::abs(dim0);
  dim1 = std::abs(dim1);
  dim2 = std::abs(dim2);

  size_t runtime_shape[3] = {static_cast<size_t>(dim0),
                             static_cast<size_t>(dim1),
                             static_cast<size_t>(dim2)};
  ynn_set_external_value_shape(runtime.get(), input_id, 3, &runtime_shape[0]);
  ynn_reshape_runtime(runtime.get());

  const size_t size = static_cast<size_t>(dim0) * dim1 * dim2;

  auto input = std::make_unique<T[]>(size);
  std::fill(input.get(), input.get() + size, static_cast<T>(1.0f));
  auto output = std::make_unique<T[]>(size);

  ynn_set_external_value_data(runtime.get(), input_id, input.get());
  ynn_set_external_value_data(runtime.get(), output_id, output.get());

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  // With an all-ones input, softmax is exactly 1 / dim2 everywhere (and the
  // plain divide is 1 everywhere).
  const float expected = plain ? 1.0f : 1.0f / dim2;
  const float tolerance = expected * 0.05f;
  if (std::any_of(output.get(), output.get() + size, [&](T x) {
        return std::abs(static_cast<float>(x) - expected) > tolerance;
      })) {
    state.SkipWithError("Incorrect result");
  }

  const size_t total_bytes = 2 * size * sizeof(T);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * total_bytes,
                                               benchmark::Counter::kIsRate);
  state.counters["OP"] = benchmark::Counter(state.iterations() * size,
                                            benchmark::Counter::kIsRate);
}

void bench(benchmark::State& state, ynn_threadpool_t threadpool, ynn_type type,
           int dim0, int dim1, int dim2, bool plain) {
  switch (type) {
    case ynn_type_fp16:
      bench<half>(state, threadpool, dim0, dim1, dim2, plain);
      break;
    case ynn_type_bf16:
      bench<bfloat16>(state, threadpool, dim0, dim1, dim2, plain);
      break;
    case ynn_type_fp32:
      bench<float>(state, threadpool, dim0, dim1, dim2, plain);
      break;
    default:
      state.SkipWithError("Unsupported type");
      break;
  }
}

}  // namespace ynn

int parse(const char* str, int) { return std::stoi(str); }

ynn_type parse(const char* str, ynn_type) {
  if (strcmp(str, "fp16") == 0) return ynn_type_fp16;
  if (strcmp(str, "bf16") == 0) return ynn_type_bf16;
  if (strcmp(str, "fp32") == 0) return ynn_type_fp32;
  return ynn_type_invalid;
}

template <typename T>
void parse_list(const char* str, std::vector<T>& result) {
  std::stringstream ss(str);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    result.push_back(parse(segment.c_str(), T{}));
  }
}

void usage(const char* name) {
  std::cout << "Usage: " << name << " [options]\n";
  std::cout << R"(
Options:
  --thread_count=N
  --type=t1,t2,...  (fp16, bf16, fp32)
  --shape=d0,d1,d2
  -d0=d0_1,d0_2,...
  -d1=d1_1,d1_2,...
  -d2=d2_1,d2_2,...
  --plain           benchmark a single standalone divide instead of the
                    fused exp/sum/divide softmax chain

Notes:
  Softmax is computed over the last (d2) axis. Multiple --type, --shape, -d0,
  -d1, and -d2 options are allowed; the Cartesian product of -d0, -d1, and
  -d2 is registered, in addition to any --shape options.

  If a shape value is positive, it is a static shape. If it is negative, it is
  a dynamic shape of the same magnitude.
)";
}

int main(int argc, char** argv) {
  constexpr unsigned max_threads = 32;
  int thread_count = std::min(max_threads, std::thread::hardware_concurrency());
  std::vector<int> dim0s;
  std::vector<int> dim1s;
  std::vector<int> dim2s;
  std::vector<std::array<int, 3>> shapes;
  std::vector<ynn_type> types;
  bool plain = false;
  benchmark::Initialize(&argc, argv);

  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "-d0=", 4) == 0) {
      parse_list(argv[i] + 4, dim0s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-d1=", 4) == 0) {
      parse_list(argv[i] + 4, dim1s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-d2=", 4) == 0) {
      parse_list(argv[i] + 4, dim2s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--shape=", 8) == 0) {
      std::vector<int> shape;
      parse_list(argv[i] + 8, shape);
      if (shape.size() != 3) {
        usage(argv[0]);
        return -1;
      }
      shapes.push_back({shape[0], shape[1], shape[2]});
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--type=", 7) == 0) {
      parse_list(argv[i] + 7, types);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--thread_count=", 15) == 0) {
      thread_count = std::stoi(argv[i] + 15);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strcmp(argv[i], "--plain") == 0) {
      plain = true;
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--benchmark_", 12) == 0 ||
               strncmp(argv[i], "-benchmark_", 11) == 0) {
      i++;
    } else {
      usage(argv[0]);
      return -1;
    }
  }

  if (types.empty()) {
    types = {ynn_type_fp32};
  }

  if (thread_count < 1) {
    usage(argv[0]);
    return -1;
  }

  if (!dim0s.empty() || !dim1s.empty() || !dim2s.empty()) {
    if (dim0s.empty()) dim0s = {1};
    if (dim1s.empty()) dim1s = {1};
    if (dim2s.empty()) dim2s = {1};
  } else if (shapes.empty()) {
    // If there is no shape specified, use a default shape.
    shapes.push_back({256, 256, 256});
  }

  ynn::TestScheduler scheduler(thread_count - 1);
  ynn::threadpool_ptr threadpool =
      ynn::create_threadpool(scheduler.scheduler(), &scheduler);

  for (ynn_type type : types) {
    std::stringstream name;
    name << (plain ? "divide_" : "softmax_") << ynn::to_string(type);
    auto* softmax_bench = benchmark::RegisterBenchmark(
        name.str(), [=, &threadpool](benchmark::State& state) {
          const int dim0 = state.range(0);
          const int dim1 = state.range(1);
          const int dim2 = state.range(2);
          ynn::bench(state, threadpool.get(), type, dim0, dim1, dim2, plain);
        });
    softmax_bench->ArgNames({"dim0", "dim1", "dim2"});
    softmax_bench->UseRealTime();
    softmax_bench->MeasureProcessCPUTime();
    for (const auto& shape : shapes) {
      softmax_bench->Args({shape[0], shape[1], shape[2]});
    }
    for (int d0 : dim0s) {
      for (int d1 : dim1s) {
        for (int d2 : dim2s) {
          softmax_bench->Args({d0, d1, d2});
        }
      }
    }
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
