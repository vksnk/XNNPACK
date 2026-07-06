load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_test_deps")

_COMPATIBLE_WITH = []

cc_library(
    name = "composites",
    srcs = [
        "attention.cc",
        "dot_quantization.cc",
        "elementwise.cc",
        "normalization.cc",
        "pooling.cc",
        "reduction.cc",
    ],
    hdrs = [
        "composites.h",
        "util.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//visibility:public"],
    deps = [
        "//ynnpack",
        "//ynnpack/base",
    ],
)

cc_test(
    name = "attention_test",
    srcs = ["attention_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
    ] + ynn_test_deps(),
)

cc_test(
    name = "attention_benchmark",
    srcs = ["attention_benchmark.cc"],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/subgraph/test:scheduler",
    ] + ynn_benchmark_deps(),
)

cc_test(
    name = "attention_parity_check",
    srcs = ["attention_parity_check.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//:XNNPACK",
        "//ynnpack",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
    ] + ynn_test_deps(),
)

cc_test(
    name = "normalization_test",
    srcs = ["normalization_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
        "//ynnpack/base",
    ] + ynn_test_deps(),
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
        "//ynnpack/base",
    ] + ynn_test_deps(),
)

cc_test(
    name = "reduction_test",
    srcs = ["reduction_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
    ] + ynn_test_deps(),
)

cc_test(
    name = "dot_quantization_test",
    srcs = ["dot_quantization_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":composites",
        "//ynnpack",
    ] + ynn_test_deps(),
)
