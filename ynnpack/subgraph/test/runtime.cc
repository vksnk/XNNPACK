// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {
// By default WASM doesn't have any thread support.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
TEST(runtime, dot_concurrency) {}
#else
TEST(runtime, dot_concurrency) {
  constexpr uint32_t a_id = 0;
  constexpr uint32_t b_id = 1;
  constexpr uint32_t c_id = 2;
  constexpr uint32_t init_zero = YNN_INVALID_VALUE_ID;

  TestScheduler scheduler(3);

  auto get_concurrency = [&](SubgraphBuilder& builder) -> int32_t {
    Runtime runtime(builder.GetSubgraph(), &scheduler);
    EXPECT_EQ(runtime.Status(), ynn_status_success);
    int32_t concurrency;
    EXPECT_EQ(runtime.Query(ynn_runtime_property_concurrency, &concurrency),
              ynn_status_success);
    return concurrency;
  };

  // We should be able to statically know this graph will not run a parallel
  // loop.
  SubgraphBuilder small(3);
  small.AddInput(type_of<float>(), {8, 8}, a_id);
  small.AddInput(type_of<float>(), {8, 8}, b_id);
  small.AddOutput(type_of<float>(), {8, 8}, c_id);
  small.AddDot(1, a_id, b_id, init_zero, c_id);
  // TODO(b/458542243): This doesn't actually work because we don't simplify
  // away these loops yet.
  // ASSERT_EQ(get_concurrency(small), 1);

  // We should be able to statically know this graph will run parallel loops.
  SubgraphBuilder big(3);
  big.AddInput(type_of<float>(), {800, 800}, a_id);
  big.AddInput(type_of<float>(), {800, 800}, b_id);
  big.AddOutput(type_of<float>(), {800, 800}, c_id);
  big.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(big), 1);

  // We don't know in this case, we might run a parallel loop if the input is
  // big enough.
  SubgraphBuilder dynamic(3);
  dynamic.AddInput(type_of<float>(), 2, a_id);
  dynamic.AddInput(type_of<float>(), 2, b_id);
  dynamic.AddOutput(type_of<float>(), 2, c_id);
  dynamic.AddDot(1, a_id, b_id, init_zero, c_id);
  ASSERT_GT(get_concurrency(dynamic), 1);
}
#endif

// Runtimes that share a cached pipeline must be able to run concurrently:
// they share the (immutable, reference counted) pipeline body, but each has
// its own constants, external buffers, and evaluation context.
TEST(runtime, pipeline_cache_concurrent_invoke) {
  constexpr uint32_t in_id = 0;
  constexpr uint32_t out_id = 1;
  constexpr size_t m = 32, k = 16;
  constexpr int num_runtimes = 4;
  constexpr int num_invokes = 20;

  std::vector<float> input(m * k);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 11) - 5.0f;
  }

  std::vector<std::unique_ptr<SubgraphBuilder>> builders;
  std::vector<std::unique_ptr<Runtime>> runtimes;
  std::vector<float> scales;
  for (int r = 0; r < num_runtimes; ++r) {
    scales.push_back(1.0f + r);
    auto builder = std::make_unique<SubgraphBuilder>(2);
    builder->AddInput(type_of<float>(), {m, k}, in_id);
    builder->AddOutput(type_of<float>(), {m, k}, out_id);
    builder->AddBinary(ynn_binary_multiply, in_id,
                       builder->DefineScalar(scales.back()), out_id);
    auto runtime = std::make_unique<Runtime>(builder->GetSubgraph());
    ASSERT_EQ(runtime->Status(), ynn_status_success);
    runtime->ReshapeExternalTensor({m, k}, input.data(), in_id)
        .ReshapeRuntime();
    builders.push_back(std::move(builder));
    runtimes.push_back(std::move(runtime));
  }

  std::vector<std::vector<float>> outputs(num_runtimes,
                                          std::vector<float>(m * k, 0.0f));
  std::vector<std::thread> threads;
  for (int r = 0; r < num_runtimes; ++r) {
    threads.emplace_back([&, r]() {
      runtimes[r]->SetupExternalTensor(outputs[r].data(), out_id);
      for (int i = 0; i < num_invokes; ++i) {
        runtimes[r]->InvokeRuntime();
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  for (int r = 0; r < num_runtimes; ++r) {
    ASSERT_EQ(runtimes[r]->Status(), ynn_status_success);
    for (size_t i = 0; i < input.size(); ++i) {
      ASSERT_EQ(outputs[r][i], input[i] * scales[r]) << "r=" << r << " i=" << i;
    }
  }
}

// Structurally identical subgraphs share a cached pipeline, with each runtime
// rebinding the pipeline's constants to its own static buffers. Run several
// such subgraphs that differ only in the contents of their constants and
// check that each one computes with its own data, not the data of the
// runtime that populated the cache.
TEST(runtime, pipeline_cache_rebinds_constants) {
  constexpr uint32_t in_id = 0;
  constexpr uint32_t out_id = 1;
  constexpr size_t m = 4, k = 8, n = 3;

  for (int trial = 0; trial < 3; ++trial) {
    std::vector<float> weights(n * k);
    float scale = 0.5f + trial;
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<float>(trial * 100 + i);
    }

    uint32_t weights_id = YNN_INVALID_VALUE_ID;
    uint32_t dot_id = YNN_INVALID_VALUE_ID;
    SubgraphBuilder builder(2);
    builder.AddInput(type_of<float>(), {m, k}, in_id);
    builder.AddOutput(type_of<float>(), {m, n}, out_id);
    builder.AddTensor(type_of<float>(), {k, n}, weights_id, weights.data());
    builder.AddTensor(type_of<float>(), {m, n}, dot_id);
    builder.AddDot(1, in_id, weights_id, YNN_INVALID_VALUE_ID, dot_id);
    builder.AddBinary(ynn_binary_multiply, dot_id, builder.DefineScalar(scale),
                      out_id);

    std::vector<float> input(m * k);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<float>(i % 7) - 3.0f;
    }
    std::vector<float> output(m * n, 0.0f);

    Runtime runtime(builder.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({m, k}, input.data(), in_id)
        .ReshapeRuntime();
    runtime.SetupExternalTensor(output.data(), out_id).InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float expected = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          expected += input[i * k + l] * weights[l * n + j];
        }
        expected *= scale;
        ASSERT_NEAR(output[i * n + j], expected, 1e-3f)
            << "trial=" << trial << " i=" << i << " j=" << j;
      }
    }
  }
}

}  // namespace ynn
