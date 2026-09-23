// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_TEST_ATTENTION_GRAPH_H_
#define XNNPACK_YNNPACK_SUBGRAPH_TEST_ATTENTION_GRAPH_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Scaled dot-product attention:
//
//   output = softmax(scale * query @ key^T) @ value
//
// `query_id` must be a [b, n, t, h] tensor, `key_id` and `value_id` must be
// [b, n, s, h] tensors; the output is [b, n, t, h] (b = batch, n = heads,
// t/s = query/key sequence length, h = head dim).
//
// If `transpose_io` is true, the inputs and output are instead sequence-major:
// query/output are [b, t, n, h] and key/value are [b, s, n, h]. The function
// inserts a {0,2,1,3} transpose on each input and on the output to convert
// to/from the head-major layout above. This mirrors what XNNPACK's attention
// subgraph does.
ynn_status define_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                            uint32_t key_id, uint32_t value_id, float scale,
                            uint32_t& output_id, bool transpose_io = false);

// Same operation as `define_attention`, but specialized for decoding.
// `query_id` must be `[b, n, 1, h]`; `key_id`/`value_id` are `[b, n, s, h]` as
// above.
//
// `define_attention` computes `S = Q @ K^T` by transposing K's last two axes
// so the (large, O(s * h)) key sequence lands as the dot's contiguous "N"
// dimension, which then has to be packed for every decode step even though
// the matmul itself only has 1 row. This version instead keeps K as the dot's
// `A` operand (its natural [.., s, h] layout already matches `A`'s contract,
// no transform needed) and makes Q the `B` operand -- Q's own last-two-axes
// transpose is then over a size-1 dimension, which is a free (aliased,
// zero-copy) or cheap. The `S = K @ Q^T` result
// lands as `[b, n, s, t]`; a second, equally free or very cheap, size-1
// transpose swaps it back to the `[b, n, t, s]` orientation the rest of the
// pipeline expects.
ynn_status define_attention_decode1(ynn_subgraph_t subgraph, uint32_t query_id,
                                    uint32_t key_id, uint32_t value_id,
                                    float scale, uint32_t& output_id,
                                    bool transpose_io = false);

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

// Same as `define_flash_attention`, but computes the transposed problem
// (S^T = K @ Q^T, U^T = V^T @ P^T) so that t is the innermost dimension of
// every per-block intermediate, and transposes the output back at the end.
ynn_status define_flash_attention_transposed(ynn_subgraph_t subgraph,
                                             uint32_t query_id, uint32_t key_id,
                                             uint32_t value_id, float scale,
                                             size_t block_width,
                                             uint32_t& output_id);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_TEST_ATTENTION_GRAPH_H_
