# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 x86 dot kernel specialized for n == 1 (a GEMV where A is the "tall"
operand): unlike every other generator in this directory, which vectorizes
along n (loading tile_n contiguous columns of B and broadcasting a scalar of
A), this one vectorizes along k and does a horizontal reduction per row of A,
so it doesn't waste SIMD lanes when n is 1 (or small). Intended for dots where
A is large (many rows) and B is a single column, e.g. attention decode's
Q @ K^T computed as K @ Q^T (see ynnpack/composites/attention.cc,
define_attention_decode1) to avoid packing K.
"""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class x86_avx2_fma3_fp32_gemv_m(dot_base):
  """tile_shape = (1, 1, 8): one row of A/output per tile, 8-wide (k)
  SIMD accumulation, reduced to a scalar once per tile in `finalize_c_tile`.
  `block_shape[0]` (> 1) unrolls multiple independent rows for ILP, since the
  FMA chain within a single row's accumulator is a true dependency."""

  def __init__(self):
    super().__init__("avx2_fma3", "fp32")
    self.a_type = "float"
    self.b_type = "float"
    self.c_type = "float"
    self.tile_shape = (1, 1, 8)

  def header(self):
    return (
        """
#include <immintrin.h>

"""
        + super().header()
    )

  def init_c_tile(self, i, j):
    return f"__m256 c_{i}_{j} = _mm256_setzero_ps();\n"

  def load_a_tile(self, i, k):
    return f"__m256 a_{i}_{k} = _mm256_loadu_ps({self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    return f"__m256 b_{k}_{j} = _mm256_loadu_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm256_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"

  def finalize_c_tile(self, i, j):
    # Horizontal-sum the 8-wide accumulator (which holds 8 independent
    # partial sums, one per SIMD lane, interleaved across k) down to the
    # single scalar dot product for this row.
    return f"""\
{{
  __m128 lo = _mm256_castps256_ps128(c_{i}_{j});
  __m128 hi = _mm256_extractf128_ps(c_{i}_{j}, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
  cs_{i}_{j} = _mm_cvtss_f32(sum1);
}}
"""

  def init_c_block(self):
    # `finalize_c_tile` writes into a plain float, declared here rather than
    # in `init_c_tile` (which must declare the __m256 accumulator).
    result = super().init_c_block()
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += f"float cs_{i}_{j};\n"
    return result

  def add_c_tile(self, i, j):
    return f"cs_{i}_{j} += *{self.c_in_ptr(i, j)};\n"

  def store_c_tile(self, i, j):
    return f"*{self.c_out_ptr(i, j)} = cs_{i}_{j};\n"

  # n is always 1 (tile_shape[1] == block_shape[1] == 1), so the "tail" case
  # (n < tile_shape[1]) never actually triggers at runtime; these just need to
  # be valid, and reusing the main-case codegen is correct since there's only
  # ever one possible n.
  def add_c_tile_tail(self, i, j, n):
    return self.add_c_tile(i, j)

  def store_c_tile_tail(self, i, j, n):
    return self.store_c_tile(i, j)


generate_dot_kernels(
    x86_avx2_fma3_fp32_gemv_m(),
    [
        (1, 1, 8),
        (2, 1, 8),
        (4, 1, 8),
        (8, 1, 8),
    ],
)
