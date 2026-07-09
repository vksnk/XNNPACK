// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                            uint32_t key_id, uint32_t value_id, float scale,
                            uint32_t& output_id, bool transpose_io) {
  // With sequence-major inputs, swap the sequence and head axes to get the
  // head-major layout the body below works in: [b, {t|s}, n, h] -> [b, n, ., h].
  const int32_t io_perm[] = {0, 2, 1, 3};
  uint32_t q_id = query_id, k_id = key_id, v_id = value_id;
  if (transpose_io) {
    q_id = k_id = v_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, query_id, &q_id, 0));
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, key_id, &k_id, 0));
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, value_id, &v_id, 0));
  }

  // S = Q @ K^T. `ynn_define_dot` contracts the last axis of `a` with the
  // second-to-last axis of `b`, so K needs its last two axes swapped.
  const int32_t k_t_perm[] = {0, 1, 3, 2};
  uint32_t k_t_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_static_transpose(subgraph, 4, k_t_perm, k_id, &k_t_id, 0));

  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, q_id, k_t_id,
                                     YNN_INVALID_VALUE_ID, &scores_id, 0));

  // P = softmax(scale * S) along the key sequence axis. softmax's `beta`
  // scaling is equivalent to scaling the scores.
  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_softmax(subgraph, scores_id, scale, probs_id));

  // O = P @ V. Head-major; transposed back to sequence-major below if needed.
  uint32_t o_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, v_id,
                                     YNN_INVALID_VALUE_ID,
                                     transpose_io ? &o_id : &output_id, 0));

  // Convert the head-major output [b, n, t, h] back to sequence-major
  // [b, t, n, h].
  if (transpose_io) {
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, o_id, &output_id, 0));
  }
  return ynn_status_success;
}

ynn_status define_attention_decode1(ynn_subgraph_t subgraph, uint32_t query_id,
                                    uint32_t key_id, uint32_t value_id,
                                    float scale, uint32_t& output_id) {
  // S = K @ Q^T (see the header comment for why this avoids packing K). Q's
  // last two axes ({..., t=1, h} -> {..., h, t=1}) need swapping so its `h`
  // axis lands in the dot's contracted (second-to-last) position; since t is
  // 1 this is a free, aliased view, not a real transpose.
  const int32_t q_t_perm[] = {0, 1, 3, 2};
  uint32_t q_t_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_static_transpose(subgraph, 4, q_t_perm, query_id, &q_t_id, 0));

  uint32_t scores_ts_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, key_id,
                                     q_t_id, YNN_INVALID_VALUE_ID,
                                     &scores_ts_id, 0));

  // scores_ts is [b, n, s, t]; swap back to [b, n, t, s] (again free, since
  // t == 1) so the rest of the pipeline is unchanged from `define_attention`.
  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(subgraph, 4, q_t_perm,
                                                   scores_ts_id, &scores_id, 0));

  // P = softmax(scale * S) along the key sequence axis.
  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_softmax(subgraph, scores_id, scale, probs_id));

  // O = P @ V. `value_id` is already `[.., s, h]`, matching `B`'s required
  // (contract=s, free=h) layout without a transpose, same as
  // `define_attention`.
  return ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, value_id,
                        YNN_INVALID_VALUE_ID, &output_id, 0);
}

ynn_status define_flash_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                                  uint32_t key_id, uint32_t value_id,
                                  float scale, size_t block_width,
                                  uint32_t& output_id) {
  if (block_width == 0) {
    return ynn_status_invalid_parameter;
  }

  // ---- Pass 1: per-block map ----
  // Scale Q up front so the block-local scores are already scaled before the
  // max reduction.
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_constant(subgraph, scale, scale_id));
  uint32_t scaled_query_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        query_id, scale_id, &scaled_query_id,
                                        0));

  // Insert the block axis into Q: [b, n, t, h] -> [b, n, 1, t, h], which
  // broadcasts Q over the blocks of K/V in the dot below.
  const int32_t block_axis_of_5d[] = {-3};
  uint32_t query_5d_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_expand_dims(
      subgraph, 1, block_axis_of_5d, scaled_query_id, &query_5d_id, 0));

  // Chop the key/value sequence into blocks of `block_width`:
  // [b, n, s, h] -> [b, n, s/w, w, h]. The number of blocks is deduced.
  const size_t splits[] = {0, block_width};
  uint32_t key_blocks_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_split_dim(subgraph, /*axis=*/-2, 2, splits,
                                           key_id, &key_blocks_id, 0));
  uint32_t value_blocks_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_split_dim(subgraph, /*axis=*/-2, 2, splits,
                                           value_id, &value_blocks_id, 0));

  // S_b = Q @ K_b^T for each block.
  const int32_t k_t_perm[] = {0, 1, 2, 4, 3};
  uint32_t key_blocks_t_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(
      subgraph, 5, k_t_perm, key_blocks_id, &key_blocks_t_id, 0));
  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, query_5d_id,
                                     key_blocks_t_id, YNN_INVALID_VALUE_ID,
                                     &scores_id, 0));

  // Block-local max m_b, un-normalized probabilities P_b = exp(S_b - m_b),
  // block-local denominator l_b and block-local output U_b = P_b @ V_b.
  const int32_t last_axis[] = {-1};
  uint32_t block_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_max, 1, last_axis,
                                        scores_id, YNN_INVALID_VALUE_ID,
                                        &block_max_id, YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t scores_minus_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        scores_id, block_max_id,
                                        &scores_minus_max_id, 0));
  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_exp,
                                       scores_minus_max_id, &probs_id, 0));
  uint32_t block_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 1, last_axis,
                                        probs_id, YNN_INVALID_VALUE_ID,
                                        &block_sum_id, YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t block_output_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id,
                                     value_blocks_id, YNN_INVALID_VALUE_ID,
                                     &block_output_id, 0));

  // Pack [m_b | l_b | U_b] along the feature axis into a single
  // [b, n, s/w, t, h + 2] tensor. The packing gives pass 1 a single consumer,
  // which lets the scheduler fuse it into one block loop and bound the
  // transient score tensors to a single block.
  const uint32_t pack_inputs[] = {block_max_id, block_sum_id, block_output_id};
  uint32_t packed_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_concatenate(subgraph, /*axis=*/-1, 3,
                                             pack_inputs, &packed_id, 0));

  // ---- Pass 2: slice the packed tensor back and combine across blocks ----
  // Plain slices are important here: they alias into `packed` instead of
  // copying it.
  const int64_t begins_m[] = {0}, ends_m[] = {1};
  const int64_t begins_l[] = {1}, ends_l[] = {2};
  // An end of 0 means the end of the axis.
  const int64_t begins_u[] = {2}, ends_u[] = {0};
  const int64_t strides[] = {1};
  uint32_t sliced_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_slice(subgraph, 1, last_axis, begins_m,
                                              ends_m, strides, packed_id,
                                              &sliced_max_id, 0));
  uint32_t sliced_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_slice(subgraph, 1, last_axis, begins_l,
                                              ends_l, strides, packed_id,
                                              &sliced_sum_id, 0));
  uint32_t local_output_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_slice(subgraph, 1, last_axis, begins_u,
                                              ends_u, strides, packed_id,
                                              &local_output_id, 0));

  // The elementwise broadcasts of m_b and l_b below must be resolvable at
  // graph construction time, but the extent of a slice is clamped by the
  // (possibly dynamic) extent of the input, so it is not provably 1.
  // Broadcasting the feature axis explicitly makes it a broadcast dimension
  // (checked to have extent 1 at runtime). This is free: it lowers to an
  // aliased transpose, and is a no-op for statically known shapes.
  uint32_t local_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_broadcast(subgraph, 1, last_axis,
                                           sliced_max_id, &local_max_id, 0));
  uint32_t local_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_broadcast(subgraph, 1, last_axis,
                                           sliced_sum_id, &local_sum_id, 0));

  // Global max over blocks, and the flash correction c_b = exp(m_b - m).
  const int32_t block_axis[] = {-3};
  uint32_t global_max_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, block_axis, local_max_id,
      YNN_INVALID_VALUE_ID, &global_max_id, YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t max_diff_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                        local_max_id, global_max_id,
                                        &max_diff_id, 0));
  uint32_t correction_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_exp, max_diff_id,
                                       &correction_id, 0));

  // Global denominator l = sum_b(c_b * l_b). The reductions below drop the
  // block axis (no KEEP_DIMS), so the final divide produces the rank 4 output
  // directly.
  uint32_t scaled_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        local_sum_id, correction_id,
                                        &scaled_sum_id, 0));
  uint32_t global_sum_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 1, block_axis,
                                        scaled_sum_id, YNN_INVALID_VALUE_ID,
                                        &global_sum_id, 0));

  // Numerator num = sum_b(c_b * U_b), and O = num / l.
  uint32_t scaled_output_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        local_output_id, correction_id,
                                        &scaled_output_id, 0));
  uint32_t numerator_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 1, block_axis,
                                        scaled_output_id, YNN_INVALID_VALUE_ID,
                                        &numerator_id, 0));
  return ynn_define_binary(subgraph, ynn_binary_divide, numerator_id,
                           global_sum_id, &output_id, 0);
}

}  // namespace ynn
