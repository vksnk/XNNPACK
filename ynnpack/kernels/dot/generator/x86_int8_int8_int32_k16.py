# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86_avx512


# In this generator, we tell the base class that we are generating 128-bit
# tiles, but we accumulate in 512-bit vectors.
class x86_avx512_int8_int8_int32_k16(x86_avx512):
  """Generates tile_k=16 avx512bw dot kernels."""

  def __init__(self, arch="avx512"):
    super().__init__(
        arch, "int8_int8_int32", "int32_t", 128, tile_shape=(1, 4, 16)
    )
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]
    # This kernel already has 2 accumulators per tile.
    self.min_tiles = max(1, self.min_tiles // 4)

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC __m512i _mm512_hadd_epi32(__m512i a, __m512i b) {
    __m512i even = _mm512_permutex2var_epi32(a, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0), b);
    __m512i odd = _mm512_permutex2var_epi32(a, _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1), b);
    return _mm512_add_epi32(even, odd);
}

}  // namespace
"""

  def b_alignment_bytes(self):
    # This kernel loads half-vectors at a time from b.
    return self.tile_shape[1] * self.tile_shape[2] // 2

  def init_c_tile(self, i, j):
    if i % 2 != 0:
      return ""
    result = f"""
__m512i c16_{i+0}_{j+0} = _mm512_setzero_si512();
__m512i c16_{i+0}_{j+2} = _mm512_setzero_si512();
__m512i c16_{i+1}_{j+0} = _mm512_setzero_si512();
__m512i c16_{i+1}_{j+2} = _mm512_setzero_si512();
"""
    return result

  # Here, we process 2 rows of 2x 512-bit accumulators, and turn them into 2
  # rows of 128-bit outputs.
  def finalize_c_tile(self, i, j):
    if i % 2 != 0:
      return ""
    result = f"""
c16_{i+0}_{j+0} = _mm512_hadd_epi32(c16_{i+0}_{j+0}, c16_{i+0}_{j+2});
c16_{i+1}_{j+0} = _mm512_hadd_epi32(c16_{i+1}_{j+0}, c16_{i+1}_{j+2});
c16_{i+0}_{j} = _mm512_hadd_epi32(c16_{i+0}_{j}, c16_{i+1}_{j});
c16_{i}_{j} = _mm512_hadd_epi32(c16_{i}_{j}, c16_{i}_{j});
__m128i c_{i+0}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 0);
"""
    if i + 1 < self.block_shape[0]:
      result += f"""
__m128i c_{i+1}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 1);
"""
    return result

  def load_a_tile(self, i, k):
    a_x16 = f"_mm_loadu_si128({self.a_ptr(i, k, '__m128i')})"
    return (
        f"__m512i a_{i}_{k} ="
        f" _mm512_cvtepi8_epi16(_mm256_broadcastsi128_si256({a_x16}));\n"
    )

  def load_b_tile(self, k, j):
    b0_ptr = self.b_ptr(k, j+0, "__m256i")
    b2_ptr = self.b_ptr(k, j+2, "__m256i")
    return f"""
__m512i b_{k}_{j+0} = _mm512_cvtepi8_epi16(_mm256_load_si256({b0_ptr}));
__m512i b_{k}_{j+2} = _mm512_cvtepi8_epi16(_mm256_load_si256({b2_ptr}));
"""

  def product(self, i, j, k):
    c_ij0 = f"c16_{i}_{j+0}"
    c_ij2 = f"c16_{i}_{j+2}"
    return f"""
{c_ij0} = _mm512_add_epi32({c_ij0}, _mm512_madd_epi16(a_{i}_{k}, b_{k}_{j+0}));
{c_ij2} = _mm512_add_epi32({c_ij2}, _mm512_madd_epi16(a_{i}_{k}, b_{k}_{j+2}));
"""


generate_dot_kernels(
    x86_avx512_int8_int8_int32_k16(),
    [
        (1, 32, 16),
        (1, 16, 16),
        (2, 16, 16),
        (1, 8, 16),
        (2, 8, 16),
        (4, 8, 16),
        (6, 8, 16),
        (2, 4, 16),
        (4, 4, 16),
        (6, 4, 16),
        (8, 4, 16),
    ],
)


class x86_avx512_int8_int8_int32_k32(dot_base):
  """int8 x int8 AVX-512BW dot kernel specialized for n == 1.

  The k-major GEMV-M counterpart of the kernels above, for machines without
  VNNI where int8 dots are not promoted to uint8: each vector holds 32 `k`
  values of one row widened to int16 (the products fit, |-128 * 127| < 2^15),
  `madd` accumulates pairwise int32 sums, and the horizontal reduction
  happens once per row at the end.
  """

  def __init__(self):
    super().__init__("avx512", "int8_int8_int32")
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.c_type = "int32_t"
    self.tile_shape = (1, 1, 32)
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return """\
#include <immintrin.h>
""" + super().header()

  def init_c_tile(self, i, j):
    return f"__m512i c_{i}_{j} = _mm512_setzero_si512();\n"

  def load_a_tile(self, i, k):
    return (
        f"__m512i a_{i}_{k} = _mm512_cvtepi8_epi16("
        f"_mm256_loadu_si256({self.a_ptr(i, k, '__m256i')}));\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m512i b_{k}_{j} = _mm512_cvtepi8_epi16("
        f"_mm256_loadu_si256({self.b_ptr(k, j, '__m256i')}));\n"
    )

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return (
        f"{c_ij} = _mm512_add_epi32("
        f"{c_ij}, _mm512_madd_epi16(a_{i}_{k}, b_{k}_{j}));\n"
    )

  def finalize_c_tile(self, i, j):
    return f"int32_t cs_{i}_{j} = _mm512_reduce_add_epi32(c_{i}_{j});\n"

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
    x86_avx512_int8_int8_int32_k32(),
    [
        (1, 1, 32),
        (2, 1, 32),
        (4, 1, 32),
        (8, 1, 32),
    ],
)
