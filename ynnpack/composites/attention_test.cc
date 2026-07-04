// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

// Layout: Q, O are [b, n, t, h]; K, V are [b, n, s, h].
struct AttentionDims {
  size_t b;  // batch
  size_t n;  // heads
  size_t t;  // query sequence length
  size_t h;  // head dim
  size_t s;  // key/value sequence length
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

using DefineAttentionFn = std::function<ynn_status(
    ynn_subgraph_t, uint32_t, uint32_t, uint32_t, float, uint32_t&)>;

void VerifyAttentionComposite(const AttentionDims& d,
                              const DefineAttentionFn& define_fn) {
  ReplicableRandomDevice rng;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d.h));

  subgraph_ptr subgraph = create_subgraph(4, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t q_id = 0, k_id = 1, v_id = 2, o_id = 3;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &q_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &k_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &v_id),
            ynn_status_success);
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 4, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &o_id),
            ynn_status_success);

  ASSERT_EQ(define_fn(subgraph.get(), q_id, k_id, v_id, scale, o_id),
            ynn_status_success);

  ASSERT_EQ(ynn_optimize_subgraph(subgraph.get(), /*threadpool=*/nullptr,
                                  /*flags=*/0),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  Tensor<float> q({d.b, d.n, d.t, d.h});
  Tensor<float> k({d.b, d.n, d.s, d.h});
  Tensor<float> v({d.b, d.n, d.s, d.h});
  fill_random(q.data(), q.size(), rng, -1.0f, 1.0f);
  fill_random(k.data(), k.size(), rng, -1.0f, 1.0f);
  fill_random(v.data(), v.size(), rng, -1.0f, 1.0f);
  Tensor<float> o({d.b, d.n, d.t, d.h});

  const size_t qo_shape[] = {d.b, d.n, d.t, d.h};
  const size_t kv_shape[] = {d.b, d.n, d.s, d.h};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), q_id, 4, qo_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), k_id, 4, kv_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), v_id, 4, kv_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), q_id, q.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), k_id, k.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), v_id, v.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), o_id, o.data()),
            ynn_status_success);

  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  Tensor<float> expected = ReferenceAttention(q, k, v, scale);
  for (const auto& i : EnumerateIndices(o.extents())) {
    ASSERT_NEAR(o(i), expected(i), 1e-3f) << "at " << index_to_string(i);
  }
}

TEST(Attention, MatchesReference) {
  VerifyAttentionComposite({/*b=*/2, /*n=*/2, /*t=*/8, /*h=*/16, /*s=*/32},
                           define_attention);
  VerifyAttentionComposite({/*b=*/1, /*n=*/3, /*t=*/5, /*h=*/16, /*s=*/64},
                           define_attention);
  VerifyAttentionComposite({/*b=*/1, /*n=*/1, /*t=*/4, /*h=*/16, /*s=*/16},
                           define_attention);
}

TEST(FlashAttention, MatchesReference) {
  auto flash_with_block_width = [](size_t block_width) {
    return [block_width](ynn_subgraph_t subgraph, uint32_t q_id, uint32_t k_id,
                         uint32_t v_id, float scale, uint32_t& o_id) {
      return define_flash_attention(subgraph, q_id, k_id, v_id, scale,
                                    block_width, o_id);
    };
  };
  VerifyAttentionComposite({/*b=*/2, /*n=*/2, /*t=*/8, /*h=*/16, /*s=*/32},
                           flash_with_block_width(8));
  VerifyAttentionComposite({/*b=*/1, /*n=*/3, /*t=*/5, /*h=*/16, /*s=*/64},
                           flash_with_block_width(16));
  VerifyAttentionComposite({/*b=*/2, /*n=*/1, /*t=*/7, /*h=*/32, /*s=*/128},
                           flash_with_block_width(64));
  // Single block: degenerates to ordinary attention.
  VerifyAttentionComposite({/*b=*/1, /*n=*/1, /*t=*/4, /*h=*/16, /*s=*/16},
                           flash_with_block_width(16));
}

}  // namespace
}  // namespace ynn
