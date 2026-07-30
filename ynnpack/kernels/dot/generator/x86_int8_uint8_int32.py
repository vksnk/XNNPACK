# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x uint8 x86 dot kernel generators.

`vpdpbusd` fixes which of its operands is the unsigned one, so a dot whose `a`
is signed and whose `b` is unsigned is a different kernel from the
`uint8_int8_int32` one, not a relabelling of it: the two vectors go into the
instruction the other way around. This combination is what a dot of two
dynamically quantized activations turns into when its operands are swapped so
that the wide one -- a key or value cache -- is the operand the dot reduces
over and therefore never has to be packed.
"""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx512vnni_int8_uint8_int32(x86_avx512):
  def __init__(self, vector_bits=512):
    super().__init__("avx512vnni", "int8_uint8_int32", "int32_t", vector_bits, (1, 16, 4))
    self.a_type = "int8_t"
    self.b_type = "uint8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC int32_t unaligned_load_s8x4(const int8_t* ptr) {
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return value;
}

}  // namespace
"""

  def load_a_tile(self, i, k):
    bits = self.bits
    a = f"unaligned_load_s8x4({self.a_ptr(i, k)})"
    return f"__m{bits}i a_{i}_{k} = _mm{self.bits}_set1_epi32({a});\n"

  def load_b_tile(self, k, j):
    bits = self.bits
    b_ptr = self.b_ptr(k, j, f"__m{bits}i")
    return f"__m{bits}i b_{k}_{j} = _mm{bits}_load_si{bits}({b_ptr});\n"

  def product(self, i, j, k):
    mm = self._mm()
    c_ij = f"c_{i}_{j}"
    # `b` is the unsigned operand here, so it goes first.
    return f"{c_ij} = {mm}_dpbusd_epi32({c_ij}, b_{k}_{j}, a_{i}_{k});\n"


generate_dot_kernels(
    x86_avx512vnni_int8_uint8_int32(),
    [
        # A dot whose operands have been swapped has the cached operand's
        # length as its rows and the query heads sharing a key head as its
        # columns, so the shapes that matter here are tall and narrow.
        (1, 16, 4),
        (2, 16, 4),
        (4, 16, 4),
        (6, 16, 4),
        (8, 16, 4),
        (10, 16, 4),
        (12, 16, 4),
        (16, 16, 4),
        (1, 32, 4),
        (2, 32, 4),
        (4, 32, 4),
        (5, 32, 4),
        (8, 32, 4),
        (1, 64, 4),
        (2, 64, 4),
        (4, 64, 4),
        (5, 64, 4),
    ],
)
