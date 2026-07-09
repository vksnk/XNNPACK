// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Verifies that the ynnpack attention graphs compute the same function as
// the xnnpack FP32Attention benchmark graph (bench/subgraph/attention.cc), so
// benchmark comparisons between the two are apples-to-apples. Replicates the
// xnn graph op-for-op (with the Q pre-scale set to 1 and the score scale set
// to 1/sqrt(h)), runs it and the ynn graphs on identical inputs, and compares
// the outputs. Layouts differ by design: the xnn graph takes seq-major
// [b, t, n, h] I/O and transposes internally; the ynn graphs take head-major
// [b, n, t, h].

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "xnnpack.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/attention_graph.h"

namespace {

struct Shape {
  size_t b, n, t, s, h, w;  // w = flash block width
};

// out[b][t][n][h], inputs indexed [b][n][seq][h].
std::vector<float> RunXnn(const Shape& sh, const std::vector<float>& q,
                          const std::vector<float>& k,
                          const std::vector<float>& v, float scale) {
  const size_t b = sh.b, n = sh.n, t = sh.t, s = sh.s, h = sh.h;
  // The xnn graph wants seq-major inputs.
  auto to_seq_major = [&](const std::vector<float>& in, size_t seq) {
    std::vector<float> out(b * seq * n * h);
    for (size_t bi = 0; bi < b; bi++)
      for (size_t ni = 0; ni < n; ni++)
        for (size_t si = 0; si < seq; si++)
          for (size_t hi = 0; hi < h; hi++)
            out[((bi * seq + si) * n + ni) * h + hi] =
                in[((bi * n + ni) * seq + si) * h + hi];
    return out;
  };
  std::vector<float> q_x = to_seq_major(q, t);
  std::vector<float> k_x = to_seq_major(k, s);
  std::vector<float> v_x = to_seq_major(v, s);
  std::vector<float> out(b * t * n * h,
                         std::numeric_limits<float>::quiet_NaN());

  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(4, 0, &subgraph) != xnn_status_success) abort();

  auto def = [&](std::initializer_list<size_t> dims, uint32_t ext_id,
                 uint32_t flags, const void* data = nullptr) {
    uint32_t id = XNN_INVALID_VALUE_ID;
    std::vector<size_t> d(dims);
    if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, d.size(),
                                d.data(), data, ext_id, flags,
                                &id) != xnn_status_success)
      abort();
    return id;
  };

  // Same values/nodes as bench/subgraph/attention.cc FP32Attention, with
  // w13 (Q pre-scale) = 1 and w15 (score scale) = `scale`.
  uint32_t v0 = def({b, s, n, h}, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);  // V
  uint32_t v1 = def({b, t, n, h}, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT);  // Q
  uint32_t v2 = def({b, s, n, h}, 2, XNN_VALUE_FLAG_EXTERNAL_INPUT);  // K
  uint32_t v3 = def({b, t, n, h}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v4 = def({b, n, t, h}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v5 = def({b, n, s, h}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v6 = def({b, n, t, s}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v7 = def({b, n, t, s}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v9 = def({b, n, t, s}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v10 = def({b, n, h, s}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v11 = def({b, n, t, h}, XNN_INVALID_VALUE_ID, 0);
  uint32_t v12 = def({b, t, n, h}, 3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);

  // Generously padded: xnn requires XNN_EXTRA_BYTES of padding after data.
  alignas(16) static float w13_data[16] = {1.0f};
  alignas(16) static float w15_data[16];
  w15_data[0] = scale;
  uint32_t w13 = def({1}, XNN_INVALID_VALUE_ID, 0, w13_data);
  uint32_t w15 = def({1}, XNN_INVALID_VALUE_ID, 0, w15_data);

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  const size_t perm0213[4] = {0, 2, 1, 3};
  const size_t perm0231[4] = {0, 2, 3, 1};
  if (xnn_define_binary(subgraph, xnn_binary_multiply, &params, v1, w13, v3,
                        0) != xnn_status_success ||
      xnn_define_static_transpose(subgraph, 4, perm0213, v3, v4, 0) !=
          xnn_status_success ||
      xnn_define_static_transpose(subgraph, 4, perm0213, v2, v5, 0) !=
          xnn_status_success ||
      xnn_define_batch_matrix_multiply(
          subgraph, v4, v5, v6, XNN_FLAG_TRANSPOSE_B | XNN_FLAG_NO_BROADCAST) !=
          xnn_status_success ||
      xnn_define_binary(subgraph, xnn_binary_multiply, &params, v6, w15, v7,
                        0) != xnn_status_success ||
      xnn_define_softmax(subgraph, v7, v9, 0) != xnn_status_success ||
      xnn_define_static_transpose(subgraph, 4, perm0231, v0, v10, 0) !=
          xnn_status_success ||
      xnn_define_batch_matrix_multiply(
          subgraph, v9, v10, v11,
          XNN_FLAG_TRANSPOSE_B | XNN_FLAG_NO_BROADCAST) !=
          xnn_status_success ||
      xnn_define_static_transpose(subgraph, 4, perm0213, v11, v12, 0) !=
          xnn_status_success)
    abort();

  xnn_runtime_t rt = nullptr;
  if (xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &rt) !=
      xnn_status_success)
    abort();
  if (xnn_reshape_runtime(rt) != xnn_status_success) abort();
  xnn_external_value externals[4] = {{0, v_x.data()},
                                     {1, q_x.data()},
                                     {2, k_x.data()},
                                     {3, out.data()}};
  if (xnn_setup_runtime_v2(rt, 4, externals) != xnn_status_success) abort();
  if (xnn_invoke_runtime(rt) != xnn_status_success) abort();
  xnn_delete_runtime(rt);
  xnn_delete_subgraph(subgraph);
  return out;
}

// out[b][n][t][h].
std::vector<float> RunComposite(const Shape& sh, const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v, float scale,
                                size_t block_width) {
  const size_t b = sh.b, n = sh.n, t = sh.t, s = sh.s, h = sh.h;
  ynn::subgraph_ptr subgraph = ynn::create_subgraph(4, 0);
  if (!subgraph) abort();

  const size_t qo_dims[4] = {b, n, t, h};
  const size_t kv_dims[4] = {b, n, s, h};
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
    status = ynn::define_attention(subgraph.get(), q_id, k_id, v_id, scale,
                                   o_id);
  } else {
    status = ynn::define_flash_attention(subgraph.get(), q_id, k_id, v_id,
                                         scale, block_width, o_id);
  }
  if (status != ynn_status_success) abort();
  if (ynn_optimize_subgraph(subgraph.get(), nullptr, 0) != ynn_status_success)
    abort();

  ynn::runtime_ptr rt = ynn::create_runtime(subgraph, nullptr, 0);
  if (!rt) abort();

  std::vector<float> out(b * n * t * h,
                         std::numeric_limits<float>::quiet_NaN());
  if (ynn_set_external_value_shape(rt.get(), q_id, 4, qo_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(rt.get(), k_id, 4, kv_dims) !=
          ynn_status_success ||
      ynn_set_external_value_shape(rt.get(), v_id, 4, kv_dims) !=
          ynn_status_success ||
      ynn_set_external_value_data(rt.get(), q_id, (void*)q.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(rt.get(), k_id, (void*)k.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(rt.get(), v_id, (void*)v.data()) !=
          ynn_status_success ||
      ynn_set_external_value_data(rt.get(), o_id, out.data()) !=
          ynn_status_success)
    abort();
  if (ynn_reshape_runtime(rt.get()) != ynn_status_success) abort();
  if (ynn_invoke_runtime(rt.get()) != ynn_status_success) abort();
  return out;
}

// Compares composite output [b][n][t][h] against xnn output [b][t][n][h].
bool Compare(const Shape& sh, const char* name,
             const std::vector<float>& composite,
             const std::vector<float>& xnn) {
  const size_t b = sh.b, n = sh.n, t = sh.t, h = sh.h;
  double max_abs = 0, max_rel = 0;
  size_t count = 0;
  for (size_t bi = 0; bi < b; bi++)
    for (size_t ni = 0; ni < n; ni++)
      for (size_t ti = 0; ti < t; ti++)
        for (size_t hi = 0; hi < h; hi++) {
          float c = composite[((bi * n + ni) * t + ti) * h + hi];
          float x = xnn[((bi * t + ti) * n + ni) * h + hi];
          double abs = std::abs((double)c - x);
          double rel = abs / std::max(1e-20, (double)std::abs(x));
          max_abs = std::max(max_abs, abs);
          max_rel = std::max(max_rel, rel);
          if (!(abs <= 1e-5 || rel <= 1e-4)) count++;
        }
  printf("%-18s b=%zu n=%zu t=%zu s=%zu h=%zu w=%zu: max_abs=%.3g "
         "max_rel=%.3g mismatches=%zu -> %s\n",
         name, sh.b, sh.n, sh.t, sh.s, sh.h, sh.w, max_abs, max_rel, count,
         count == 0 ? "OK" : "FAIL");
  return count == 0;
}

}  // namespace

int main() {
  if (xnn_initialize(nullptr) != xnn_status_success) abort();

  const Shape shapes[] = {
      {2, 3, 17, 32, 8, 16},
      {1, 4, 64, 64, 32, 32},
      {1, 2, 100, 256, 64, 64},
  };

  bool ok = true;
  for (const Shape& sh : shapes) {
    const float scale = 1.0f / std::sqrt((float)sh.h);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(sh.b * sh.n * sh.t * sh.h);
    std::vector<float> k(sh.b * sh.n * sh.s * sh.h);
    std::vector<float> v(sh.b * sh.n * sh.s * sh.h);
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);

    std::vector<float> ref = RunXnn(sh, q, k, v, scale);
    ok &= Compare(sh, "attention", RunComposite(sh, q, k, v, scale, 0), ref);
    ok &= Compare(sh, "flash_attention",
                  RunComposite(sh, q, k, v, scale, sh.w), ref);
  }
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
