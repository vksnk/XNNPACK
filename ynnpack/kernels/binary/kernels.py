"""Definition of binary kernels."""

# pylint: disable=undefined-variable
# pylint: disable=g-unsafe-pickle-load
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("add")
def add_fp32(a, b, x):
  return store(load(a) + load(b), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("add")
def add_fp64(a, b, x):
  return store(load(a) + load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("subtract")
def subtract_fp32(a, b, x):
  return store(load(a) - load(b), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("subtract")
def subtract_fp64(a, b, x):
  return store(load(a) - load(b), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("subtract")
def subtract_bf16_fp32(a, b, x):
  return store(cast(Float(32), load(a)) - load(b), x)


@const_buffer("a", Float(32))
@const_buffer("b", BFloat(16))
@buffer("x", BFloat(16))
@operator_name("subtract")
def subtract_fp32_bf16_bf16(a, b, x):
  return store(cast(BFloat(16), load(a) - cast(Float(32), load(b))), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("exp_subtract")
def exp_subtract_fp32(a, b, x):
  return store(exp(load(a) - load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_fp32(a, b, x):
  return store(load(a) * load(b), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("multiply")
def multiply_fp64(a, b, x):
  return store(load(a) * load(b), x)


@const_buffer("a", Int(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_int32_fp32(a, b, x):
  return store(cast(Float(32), load(a)) * load(b), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("multiply")
def multiply_bf16_fp32(a, b, x):
  return store(cast(Float(32), load(a)) * load(b), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", Float(32))
@buffer("x", BFloat(16))
@operator_name("multiply")
def multiply_bf16_fp32_bf16(a, b, x):
  return store(cast(BFloat(16), cast(Float(32), load(a)) * load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("divide")
def divide_fp32(a, b, x):
  return store(load(a) / load(b), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("divide")
def divide_fp64(a, b, x):
  return store(load(a) / load(b), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("divide")
def divide_bf16_fp32(a, b, x):
  return store(cast(Float(32), load(a)) / load(b), x)


@const_buffer("a", BFloat(16))
@const_buffer("b", Float(32))
@buffer("x", BFloat(16))
@operator_name("divide")
def divide_bf16_fp32_bf16(a, b, x):
  return store(cast(BFloat(16), cast(Float(32), load(a)) / load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("max")
def max_fp32(a, b, x):
  return store(max(load(a), load(b)), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("max")
def max_fp64(a, b, x):
  return store(max(load(a), load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("min")
def min_fp32(a, b, x):
  return store(min(load(a), load(b)), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("min")
def min_fp64(a, b, x):
  return store(min(load(a), load(b)), x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("squared_difference")
def squared_difference_fp32(a, b, x):
  diff = load(a) - load(b)
  return store(diff * diff, x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("squared_difference")
def squared_difference_fp64(a, b, x):
  diff = load(a) - load(b)
  return store(diff * diff, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", Float(32))
@operator_name("copysign")
def copysign_fp32(a, b, x):
  va = load(a)
  vb = load(b)
  # TODO (vksnk): we shouldn't need this cast if we add patterns for bit binary
  # ops which do reinterpret_cast themselves.
  mask = reinterpret_cast(Float(32), 0x7FFFFFFF)
  return store(select_bits(mask, va, vb), x)


@const_buffer("a", Float(64))
@const_buffer("b", Float(64))
@buffer("x", Float(64))
@operator_name("copysign")
def copysign_fp64(a, b, x):
  va = load(a)
  vb = load(b)
  mask = reinterpret_cast(Float(64), 0x7FFFFFFFFFFFFFFF)
  return store(select_bits(mask, va, vb), x)


# Comparisons produce a uint8 0 or 1, which is how the delegates represent a
# boolean tensor (and what `select` expects as its condition).
def _compare_fp32(cmp):
  def make(a, b, x):
    # Select in fp32 and narrow with the float -> uint8 conversion, which is
    # the lowering the SIMD backends implement for every architecture.
    one = select(cmp(load(a), load(b)), 1.0, 0.0)
    return store(cast(UInt(8), one), x)

  return make


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("equal")
def equal_fp32(a, b, x):
  return _compare_fp32(equal)(a, b, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("not_equal")
def not_equal_fp32(a, b, x):
  return _compare_fp32(not_equal)(a, b, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("less")
def less_fp32(a, b, x):
  return _compare_fp32(lambda p, q: p < q)(a, b, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("less_equal")
def less_equal_fp32(a, b, x):
  return _compare_fp32(lambda p, q: p <= q)(a, b, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("greater")
def greater_fp32(a, b, x):
  return _compare_fp32(lambda p, q: p > q)(a, b, x)


@const_buffer("a", Float(32))
@const_buffer("b", Float(32))
@buffer("x", UInt(8))
@operator_name("greater_equal")
def greater_equal_fp32(a, b, x):
  return _compare_fp32(lambda p, q: p >= q)(a, b, x)


# Logical ops on uint8 tensors, where any non-zero value is true and the
# result is 0 or 1.
# Normalizing to 0.0/1.0 in fp32 and narrowing with the float -> uint8
# conversion reuses the lowering every architecture implements; `min`/`max` of
# those are then `and`/`or`.
def _as_bool(v):
  return select(not_equal(cast(Float(32), v), 0.0), 1.0, 0.0)


@const_buffer("a", UInt(8))
@const_buffer("b", UInt(8))
@buffer("x", UInt(8))
@operator_name("logical_and")
def logical_and_uint8(a, b, x):
  return store(cast(UInt(8), min(_as_bool(load(a)), _as_bool(load(b)))), x)


@const_buffer("a", UInt(8))
@const_buffer("b", UInt(8))
@buffer("x", UInt(8))
@operator_name("logical_or")
def logical_or_uint8(a, b, x):
  return store(cast(UInt(8), max(_as_bool(load(a)), _as_bool(load(b)))), x)
