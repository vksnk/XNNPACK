// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/binary/reference.h"

#include "ynnpack/include/ynnpack.h"

namespace ynn {

const binary_op_info* get_binary_op_info(ynn_binary_operator op) {
  static add add;
  static copysign copysign;
  static divide div;
  static max max;
  static min min;
  static multiply mul;
  static struct pow pow;
  static squared_difference squared_difference;
  static subtract sub;
  static leaky_relu leaky_relu;
  static exp_subtract exp_subtract;
  static struct equal equal;
  static struct not_equal not_equal;
  static struct less less;
  static struct less_equal less_equal;
  static struct greater greater;
  static struct greater_equal greater_equal;
  static struct logical_and logical_and;
  static struct logical_or logical_or;

  switch (op) {
    case ynn_binary_add:
      return &add;
    case ynn_binary_copysign:
      return &copysign;
    case ynn_binary_divide:
      return &div;
    case ynn_binary_max:
      return &max;
    case ynn_binary_min:
      return &min;
    case ynn_binary_multiply:
      return &mul;
    case ynn_binary_pow:
      return &pow;
    case ynn_binary_squared_difference:
      return &squared_difference;
    case ynn_binary_subtract:
      return &sub;
    case ynn_binary_leaky_relu:
      return &leaky_relu;
    case ynn_binary_exp_subtract:
      return &exp_subtract;
    case ynn_binary_equal:
      return &equal;
    case ynn_binary_not_equal:
      return &not_equal;
    case ynn_binary_less:
      return &less;
    case ynn_binary_less_equal:
      return &less_equal;
    case ynn_binary_greater:
      return &greater;
    case ynn_binary_greater_equal:
      return &greater_equal;
    case ynn_binary_logical_and:
      return &logical_and;
    case ynn_binary_logical_or:
      return &logical_or;
    case ynn_binary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // namespace ynn
