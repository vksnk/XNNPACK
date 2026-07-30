// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

constexpr size_t kSize = 129;  // Not a multiple of any vector width.

// out = condition ? a : b, for values of type T.
template <typename T>
void TestSelect(bool broadcast_condition) {
  const uint32_t condition_id = 0;
  const uint32_t a_id = 1;
  const uint32_t b_id = 2;
  const uint32_t out_id = 3;

  const std::vector<size_t> shape = {kSize};
  const std::vector<size_t> condition_shape =
      broadcast_condition ? std::vector<size_t>{1} : shape;

  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_uint8, condition_shape, condition_id)
      .AddInput(type_of<T>(), shape, a_id)
      .AddInput(type_of<T>(), shape, b_id)
      .AddOutput(type_of<T>(), shape, out_id);

  uint32_t out = out_id;
  ASSERT_EQ(ynn_define_select(builder.GetSubgraph(), condition_id, a_id, b_id,
                              &out, /*flags=*/0),
            ynn_status_success);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<uint8_t> condition(condition_shape);
  Tensor<T> a(shape);
  Tensor<T> b(shape);
  Tensor<T> out_data(shape);
  for (size_t i = 0; i < condition.size(); ++i) {
    // Use a value other than 1 for true sometimes: any non-zero is true.
    condition(i) = (i % 3 == 0) ? 0 : static_cast<uint8_t>(i % 5 + 1);
  }
  for (size_t i = 0; i < kSize; ++i) {
    a(i) = static_cast<T>(i);
    b(i) = static_cast<T>(1000 + i);
  }

  runtime.ReshapeExternalTensor(condition_shape, condition.base(), condition_id)
      .ReshapeExternalTensor(shape, a.base(), a_id)
      .ReshapeExternalTensor(shape, b.base(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out_data.base(), out_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < kSize; ++i) {
    const uint8_t c = broadcast_condition ? condition(0) : condition(i);
    EXPECT_EQ(out_data(i), c ? a(i) : b(i)) << "i=" << i;
  }
}

TEST(select, fp32) { TestSelect<float>(/*broadcast_condition=*/false); }
TEST(select, int32) { TestSelect<int32_t>(/*broadcast_condition=*/false); }
TEST(select, broadcast_condition) {
  TestSelect<float>(/*broadcast_condition=*/true);
}

TEST(select, rejects_non_uint8_condition) {
  const uint32_t condition_id = 0;
  const uint32_t a_id = 1;
  const uint32_t b_id = 2;
  const uint32_t out_id = 3;

  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_fp32, {kSize}, condition_id)
      .AddInput(ynn_type_fp32, {kSize}, a_id)
      .AddInput(ynn_type_fp32, {kSize}, b_id)
      .AddOutput(ynn_type_fp32, {kSize}, out_id);

  uint32_t out = out_id;
  EXPECT_EQ(ynn_define_select(builder.GetSubgraph(), condition_id, a_id, b_id,
                              &out, /*flags=*/0),
            ynn_status_unsupported_parameter);
}

// Comparisons write a 0 or 1 uint8 regardless of the input type.
void TestCompare(ynn_binary_operator op) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t out_id = 2;

  const std::vector<size_t> shape = {kSize};
  SubgraphBuilder builder(3);
  builder.AddInput(ynn_type_fp32, shape, a_id)
      .AddInput(ynn_type_fp32, shape, b_id)
      .AddOutput(ynn_type_uint8, shape, out_id);

  uint32_t out = out_id;
  ASSERT_EQ(
      ynn_define_binary(builder.GetSubgraph(), op, a_id, b_id, &out, /*flags=*/0),
      ynn_status_success);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> a(shape);
  Tensor<float> b(shape);
  Tensor<uint8_t> out_data(shape);
  for (size_t i = 0; i < kSize; ++i) {
    a(i) = static_cast<float>(i % 7) - 3.0f;
    b(i) = static_cast<float>(i % 5) - 2.0f;
  }

  runtime.ReshapeExternalTensor(shape, a.base(), a_id)
      .ReshapeExternalTensor(shape, b.base(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out_data.base(), out_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < kSize; ++i) {
    bool expected = false;
    switch (op) {
      case ynn_binary_equal:
        expected = a(i) == b(i);
        break;
      case ynn_binary_not_equal:
        expected = a(i) != b(i);
        break;
      case ynn_binary_less:
        expected = a(i) < b(i);
        break;
      case ynn_binary_less_equal:
        expected = a(i) <= b(i);
        break;
      case ynn_binary_greater:
        expected = a(i) > b(i);
        break;
      case ynn_binary_greater_equal:
        expected = a(i) >= b(i);
        break;
      default:
        FAIL() << "unexpected operator";
    }
    EXPECT_EQ(out_data(i), expected ? 1 : 0) << "i=" << i << " op=" << op;
  }
}

TEST(compare, equal) { TestCompare(ynn_binary_equal); }
TEST(compare, not_equal) { TestCompare(ynn_binary_not_equal); }
TEST(compare, less) { TestCompare(ynn_binary_less); }
TEST(compare, less_equal) { TestCompare(ynn_binary_less_equal); }
TEST(compare, greater) { TestCompare(ynn_binary_greater); }
TEST(compare, greater_equal) { TestCompare(ynn_binary_greater_equal); }

// The pattern the models use: mask = a != b; out = mask ? x : y.
TEST(select, of_comparison) {
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  const uint32_t y_id = 2;
  const uint32_t out_id = 3;

  const std::vector<size_t> shape = {kSize};
  SubgraphBuilder builder(4);
  builder.AddInput(ynn_type_fp32, shape, a_id)
      .AddInput(ynn_type_fp32, shape, x_id)
      .AddInput(ynn_type_fp32, shape, y_id)
      .AddOutput(ynn_type_fp32, shape, out_id);

  const float pad_value = 7.0f;
  uint32_t pad_id = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, {}, pad_id, &pad_value,
                    YNN_VALUE_FLAG_COPY_DATA);

  uint32_t mask_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_binary(builder.GetSubgraph(), ynn_binary_not_equal, a_id,
                              pad_id, &mask_id, /*flags=*/0),
            ynn_status_success);

  uint32_t out = out_id;
  ASSERT_EQ(ynn_define_select(builder.GetSubgraph(), mask_id, x_id, y_id, &out,
                              /*flags=*/0),
            ynn_status_success);

  Runtime runtime(builder.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> a(shape);
  Tensor<float> x(shape);
  Tensor<float> y(shape);
  Tensor<float> out_data(shape);
  for (size_t i = 0; i < kSize; ++i) {
    a(i) = (i % 4 == 0) ? pad_value : static_cast<float>(i);
    x(i) = static_cast<float>(i);
    y(i) = -1.0f;
  }

  runtime.ReshapeExternalTensor(shape, a.base(), a_id)
      .ReshapeExternalTensor(shape, x.base(), x_id)
      .ReshapeExternalTensor(shape, y.base(), y_id)
      .ReshapeRuntime()
      .SetupExternalTensor(out_data.base(), out_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(out_data(i), a(i) != pad_value ? x(i) : y(i)) << "i=" << i;
  }
}

}  // namespace
}  // namespace ynn
