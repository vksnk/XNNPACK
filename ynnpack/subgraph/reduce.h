// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_
#define XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_

#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"

namespace ynn {

float get_reduce_identity(ynn_reduce_operator op);

// Makes an identity for a reduction. For most reductions, this is a simple
// rank-zero scalar. But for min_max reductions, we need a different reduction
// identity value for the min and the max.
slinky::raw_buffer_ptr make_reduce_identity(ynn_type type, int rank,
                                            ynn_reduce_operator op);

void define_reduce(ynn_subgraph& subgraph, ynn_node& node,
                   ynn_reduce_operator op, const ynn::axes_set& k_dims,
                   uint32_t input_a_id, uint32_t input_b_id,
                   uint32_t* output_id, bool keep_dims,
                   std::vector<slinky::expr> split_factors = {});

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_REDUCE_H_
