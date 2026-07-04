// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
#define XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// This header defines helpers for implementing common higher level operations
// using YNNPACK's lower level public API.

// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
ynn_status define_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                       uint32_t& output_id);

// approx_gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
ynn_status define_approx_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id);

// elu(x) = x >= 0 ? x : alpha * (exp(x) - 1)
ynn_status define_elu(ynn_subgraph_t subgraph, uint32_t input_id, float alpha,
                      uint32_t& output_id);

// leaky_relu(x) = x >= 0 ? x : alpha * x
ynn_status define_leaky_relu(ynn_subgraph_t subgraph, uint32_t input_id,
                             float alpha, uint32_t& output_id);

// hardswish(x) = x * clamp(x / 6 + 0.5, 0, 1)
ynn_status define_hardswish(ynn_subgraph_t subgraph, uint32_t input_id,
                            uint32_t& output_id);

// softmax(x)_i = exp(beta * x_i) / sum_j(exp(beta * x_j))
ynn_status define_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                          float beta, uint32_t& output_id);

// log_softmax(x)_i = x_i - log(sum_j(exp(x_j)))
ynn_status define_log_softmax(ynn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t& output_id);

// Computes average pooling of a 2D buffer. The `input_id` and `output_id`
// values must refer to rank 4 tensors. When averaging, the number of samples is
// the number of samples that are not padding.
ynn_status define_average_pool_2d(ynn_subgraph_t subgraph, uint32_t input_id,
                                  ynn_type type, bool padding_same,
                                  size_t filter_height, size_t filter_width,
                                  size_t stride_height, size_t stride_width,
                                  uint32_t& output_id);

// Computes a sum reduction, optionally dividing by the number of elements in
// the reduction if `mean` is true. The quantization parameters may be
// `YNN_INVALID_VALUE_ID`, indicating an identity value of 0 (for zero point) or
// 1 (for scale).
ynn_status define_reduce_sum(ynn_subgraph_t subgraph, size_t num_axes,
                             const int32_t* axes, uint32_t input_id,
                             uint32_t input_zero_point_id,
                             uint32_t input_scale_id, bool keep_dims, bool mean,
                             bool squared, ynn_type output_type,
                             uint32_t output_zero_point_id,
                             uint32_t output_scale_id, uint32_t& output_id);

// Scaled dot-product attention:
//
//   output = softmax(scale * query @ key^T) @ value
//
// `query_id` must be a [b, n, t, h] tensor, `key_id` and `value_id` must be
// [b, n, s, h] tensors; the output is [b, n, t, h] (b = batch, n = heads,
// t/s = query/key sequence length, h = head dim). The softmax is computed over
// the full key sequence, so the transient score tensors are O(t * s) per head.
ynn_status define_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                            uint32_t key_id, uint32_t value_id, float scale,
                            uint32_t& output_id);

// Computes the same operation as `define_attention` using a memory-efficient
// ("flash attention") two-pass rfactor. The key/value sequence is chopped into
// blocks of `block_width` (which must divide s); pass 1 computes a block-local
// softmax and output, packed into a single [b, n, s/w, t, h + 2] tensor; pass 2
// rescales the blocks by exp(m_block - m_global) and combines them. The packing
// lets the scheduler fuse pass 1 into one block loop, bounding the transient
// score tensors to O(t * block_width) per head instead of O(t * s).
ynn_status define_flash_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                                  uint32_t key_id, uint32_t value_id,
                                  float scale, size_t block_width,
                                  uint32_t& output_id);

// This function computes the quantization parameters of the result of a
// quantized dot operation. It computes the `zero_point` and `scale` values of
// the following equivalence:
//
//   (a.b - zero_point)*scale =
//       ((a - a_zero_point)*a_scale).(b - b_zerp_point)*b_scale
//
// It supports dynamic and static quantization parameters.
ynn_status define_dot_quantization(ynn_subgraph_t subgraph, size_t num_k_dims,
                                   uint32_t a_id, uint32_t a_zero_point_id,
                                   uint32_t a_scale_id, uint32_t b_id,
                                   uint32_t b_zero_point_id,
                                   uint32_t b_scale_id, uint32_t& zero_point_id,
                                   uint32_t& scale_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_COMPOSITES_COMPOSITES_H_
