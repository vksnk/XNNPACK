"""Definition of ternary kernels."""

# pylint: disable=undefined-variable
# pylint: disable=g-unsafe-pickle-load
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_fp32_fp32_fp32(a, b, c, x):
  return store(load(a) * load(b) * load(c), x)


@const_buffer("a", Int(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_int32_fp32_fp32(a, b, c, x):
  return store(cast(Float(32), load(a)) * load(b) * load(c), x)


@const_buffer("a", Int(32))
@const_buffer("b", Int(32))
@const_buffer("c", Int(32))
@buffer("x", Int(32))
@operator_name("subtract_multiply")
def subtract_multiply_int32_int32_int32(a, b, c, x):
  return store(load(a) - load(b) * load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("multiply_add")
def multiply_add_fp32_fp32_fp32(a, b, c, x):
  return store(load(a) * load(b) + load(c), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@const_buffer("c", Float(64))
@buffer("x", Float(64))
@operator_name("multiply_add")
def multiply_add_fp64_fp64_fp64(a, b, c, x):
  return store(load(a) * load(b) + load(c), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("clamp")
def clamp_fp32_fp32_fp32(a, b, c, x):
  return store(min(max(load(a), load(b)), load(c)), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@const_buffer("c", Float(64))
@buffer("x", Float(64))
@operator_name("clamp")
def clamp_fp64_fp64_fp64(a, b, c, x):
  return store(min(max(load(a), load(b)), load(c)), x)


# `select` takes the condition as uint8 (0 or 1), which is how the delegates
# represent boolean tensors. The condition is converted to the type of the
# values first so all the lanes have the same width.
@const_buffer("a", UInt(8))
@const_buffer("b", Float(32))
@const_buffer("c", Float(32))
@buffer("x", Float(32))
@operator_name("select")
def select_uint8_fp32_fp32(a, b, c, x):
  cond = cast(Float(32), load(a))
  return store(select(not_equal(cond, 0.0), load(b), load(c)), x)


@const_buffer("a", UInt(8))
@const_buffer("b", Int(32))
@const_buffer("c", Int(32))
@buffer("x", Int(32))
@operator_name("select")
def select_uint8_int32_int32(a, b, c, x):
  cond = cast(Int(32), load(a))
  return store(select(not_equal(cond, 0), load(b), load(c)), x)
