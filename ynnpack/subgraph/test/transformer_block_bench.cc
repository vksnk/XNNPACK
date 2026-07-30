// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This benchmark reconstructs the tail of a gemma3 transformer layer (the
// attention output projection, the feed forward network, and the RMS
// normalizations around them) as observed in slinky IR dumps of a real model.
// All of the matmuls are dynamically quantized (fp32 activations quantized to
// int8 per token, statically per-channel quantized int8 weights), which is
// what litert-lm uses. Logically, all tensors are [tokens, channels] (channels
// innermost).
//
// It reproduces a pathological schedule: the layer output is an external
// output of the graph which the trailing normalization also consumes. Slinky
// never allocates the pipeline's output buffers, so it also never crops them
// to the region a loop iteration needs; when the scheduler fuses the producer
// of the layer output into the trailing normalization's parallel loop over
// tokens, the whole block is recomputed for *all* tokens on every iteration of
// that loop. With 1024 tokens and a step of 64 that is 16x the work.
//
// The `external_residual` and `final_norm` arguments are ablations: making the
// layer output an internal value lets slinky crop it, and dropping the
// trailing normalization removes the loop to fuse into. Both run at the speed
// the un-fused block should have.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

// Shapes of the "tiny" gemma3 model the IR dumps came from.
constexpr size_t kHidden = 640;
constexpr size_t kFfn = 2048;
constexpr size_t kHeads = 4;
constexpr size_t kHeadDim = 256;
constexpr size_t kAttnOut = kHeads * kHeadDim;  // 1024
constexpr float kEpsilon = 1e-6f;

#define RETURN_IF_ERROR(x)                       \
  do {                                           \
    ynn_status status_ = (x);                    \
    if (status_ != ynn_status_success) {         \
      return status_;                            \
    }                                            \
  } while (0)

// Holds the (fake) weights of the block. The subgraph refers to this data
// without copying it, so it must outlive the runtime.
struct BlockWeights {
  std::vector<int8_t> w_o;      // [kAttnOut, kHidden]
  std::vector<float> s_o;       // [kHidden]
  std::vector<int8_t> w_gate;   // [kHidden, kFfn]
  std::vector<float> s_gate;    // [kFfn]
  std::vector<int8_t> w_up;     // [kHidden, kFfn]
  std::vector<float> s_up;      // [kFfn]
  std::vector<int8_t> w_down;   // [kFfn, kHidden]
  std::vector<float> s_down;    // [kHidden]
  std::vector<float> gamma[4];  // [kHidden]

  BlockWeights() {
    auto fill_weights = [](std::vector<int8_t>& w, size_t n) {
      w.resize(n);
      for (size_t i = 0; i < n; ++i) {
        w[i] = static_cast<int8_t>((i % 15) - 7);
      }
    };
    auto fill_scales = [](std::vector<float>& s, size_t n) {
      s.assign(n, 0.01f);
    };
    fill_weights(w_o, kAttnOut * kHidden);
    fill_scales(s_o, kHidden);
    fill_weights(w_gate, kHidden * kFfn);
    fill_scales(s_gate, kFfn);
    fill_weights(w_up, kHidden * kFfn);
    fill_scales(s_up, kFfn);
    fill_weights(w_down, kFfn * kHidden);
    fill_scales(s_down, kHidden);
    for (std::vector<float>& g : gamma) g.assign(kHidden, 1.0f);
  }
};

class BlockBuilder {
 public:
  explicit BlockBuilder(ynn_subgraph_t subgraph) : subgraph_(subgraph) {}

  uint32_t Constant(ynn_type type, const std::vector<size_t>& dims,
                    const void* data) {
    uint32_t id = YNN_INVALID_VALUE_ID;
    ynn_define_tensor(subgraph_, type, dims.size(), dims.data(), data,
                      /*flags=*/0, &id);
    return id;
  }

  uint32_t Scalar(float value) {
    uint32_t id = YNN_INVALID_VALUE_ID;
    ynn_define_tensor(subgraph_, ynn_type_fp32, 0, nullptr, &value,
                      /*flags=*/YNN_VALUE_FLAG_COPY_DATA, &id);
    return id;
  }

  // out = gamma * x * rsqrt(mean(x^2) + epsilon), reducing over the last
  // (channel) dimension of `x`.
  ynn_status RmsNorm(uint32_t x_id, const std::vector<float>& gamma,
                     size_t channels, uint32_t& out_id) {
    const int32_t channel_axis[] = {-1};

    uint32_t sum_squared_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_reduce(subgraph_, ynn_reduce_sum_squared,
                                      /*num_axes=*/1, channel_axis, x_id,
                                      YNN_INVALID_VALUE_ID, &sum_squared_id,
                                      /*flags=*/0));

    uint32_t mean_squared_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_binary(
        subgraph_, ynn_binary_multiply, sum_squared_id,
        Scalar(1.0f / static_cast<float>(channels)), &mean_squared_id, 0));

    uint32_t biased_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_binary(subgraph_, ynn_binary_add,
                                      mean_squared_id, Scalar(kEpsilon),
                                      &biased_id, 0));

    // Restore the reduced dimension so the result broadcasts over channels.
    uint32_t expanded_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_static_expand_dims(
        subgraph_, /*num_new_axes=*/1, channel_axis, biased_id, &expanded_id,
        /*flags=*/0));

    uint32_t inv_rms_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_unary(subgraph_, ynn_unary_rsqrt, expanded_id,
                                     &inv_rms_id, 0));

    uint32_t normalized_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_binary(subgraph_, ynn_binary_multiply, x_id,
                                      inv_rms_id, &normalized_id, 0));

    return ynn_define_binary(subgraph_, ynn_binary_multiply, normalized_id,
                             Constant(ynn_type_fp32, {channels}, gamma.data()),
                             &out_id, 0);
  }

  // Quantizes `x_id` to int8, with a scale and zero point computed per token
  // from the min/max over the channel dimension.
  ynn_status DynamicQuantize(uint32_t x_id, uint32_t& quantized_id,
                             uint32_t& zero_point_id, uint32_t& scale_id) {
    const int32_t channel_axis[] = {-1};
    uint32_t min_max_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_reduce(subgraph_, ynn_reduce_min_max,
                                      /*num_axes=*/1, channel_axis, x_id,
                                      YNN_INVALID_VALUE_ID, &min_max_id,
                                      YNN_NODE_FLAG_KEEP_DIMS));

    zero_point_id = YNN_INVALID_VALUE_ID;
    scale_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_dynamic_quantization(
        subgraph_, min_max_id, ynn_type_int8, &zero_point_id, &scale_id,
        /*flags=*/0));

    quantized_id = YNN_INVALID_VALUE_ID;
    return ynn_define_quantize(subgraph_, x_id, ynn_type_int8, zero_point_id,
                               scale_id, &quantized_id, /*flags=*/0);
  }

  // A matmul of a dynamically quantized activation with per-channel quantized
  // int8 weights, dequantized back to fp32. `w` is [k, n], `s` is [n].
  ynn_status QuantizedDot(uint32_t a_id, uint32_t a_zero_point_id,
                          uint32_t a_scale_id, const std::vector<int8_t>& w,
                          const std::vector<float>& s, size_t k, size_t n,
                          uint32_t& out_id) {
    const uint32_t w_id = Constant(ynn_type_int8, {k, n}, w.data());
    const uint32_t s_id = Constant(ynn_type_fp32, {n}, s.data());

    uint32_t dot_zero_point_id = YNN_INVALID_VALUE_ID;
    uint32_t dot_scale_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(define_dot_quantization(
        subgraph_, /*num_k_dims=*/1, a_id, a_zero_point_id, a_scale_id, w_id,
        /*b_zero_point_id=*/YNN_INVALID_VALUE_ID, s_id, dot_zero_point_id,
        dot_scale_id));

    uint32_t accumulator_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_dot(subgraph_, /*num_k_dims=*/1, a_id, w_id,
                                   /*input_c_id=*/YNN_INVALID_VALUE_ID,
                                   &accumulator_id, /*flags=*/0));

    return ynn_define_dequantize(subgraph_, accumulator_id, dot_zero_point_id,
                                 dot_scale_id, ynn_type_fp32, &out_id,
                                 /*flags=*/0);
  }

 private:
  ynn_subgraph_t subgraph_;
};

// Builds:
//
//   attn = transpose(attn_in)                      # [heads, t, head] -> [t, c]
//   h    = residual + rms_norm(attn @ W_o)
//   x    = rms_norm(h)
//   ffn  = (gelu(x @ W_gate) * (x @ W_up)) @ W_down
//   out  = h + rms_norm(ffn)                       # external output
//   norm = rms_norm(out)                           # external output
ynn_status DefineBlock(ynn_subgraph_t subgraph, const BlockWeights& weights,
                       size_t tokens, bool static_shape, bool transpose_attn,
                       bool external_residual_output, bool final_norm,
                       uint32_t attn_id, uint32_t residual_id,
                       uint32_t block_out_id, uint32_t norm_out_id) {
  BlockBuilder b(subgraph);

  const size_t attn_dims[] = {kHeads, tokens, kHeadDim};
  const size_t attn_flat_dims[] = {tokens, kAttnOut};
  const size_t hidden_dims[] = {tokens, kHidden};

  RETURN_IF_ERROR(ynn_define_tensor(
      subgraph, ynn_type_fp32, transpose_attn ? 3 : 2,
      static_shape ? (transpose_attn ? attn_dims : attn_flat_dims) : nullptr,
      nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &attn_id));
  RETURN_IF_ERROR(ynn_define_tensor(subgraph, ynn_type_fp32, 2,
                                    static_shape ? hidden_dims : nullptr,
                                    nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &residual_id));
  RETURN_IF_ERROR(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr,
                                    nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &norm_out_id));

  uint32_t attn_flat_id = attn_id;
  if (transpose_attn) {
    // The attention output is head major; the projection consumes it as
    // [tokens, heads * head_dim].
    const int32_t perm[] = {1, 0, 2};
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_static_transpose(subgraph, /*rank=*/3, perm,
                                                attn_id, &transposed_id,
                                                /*flags=*/0));
    attn_flat_id = YNN_INVALID_VALUE_ID;
    RETURN_IF_ERROR(ynn_define_fuse_dim(subgraph, /*axis=*/1, /*axes_count=*/2,
                                        transposed_id, &attn_flat_id,
                                        /*flags=*/0));
  }

  // Output projection of the attention result.
  uint32_t attn_q_id, attn_zp_id, attn_scale_id;
  RETURN_IF_ERROR(
      b.DynamicQuantize(attn_flat_id, attn_q_id, attn_zp_id, attn_scale_id));
  uint32_t projected_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(b.QuantizedDot(attn_q_id, attn_zp_id, attn_scale_id,
                                 weights.w_o, weights.s_o, kAttnOut, kHidden,
                                 projected_id));

  uint32_t post_attn_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(
      b.RmsNorm(projected_id, weights.gamma[0], kHidden, post_attn_id));

  uint32_t residual_sum_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add, residual_id,
                                    post_attn_id, &residual_sum_id, 0));

  // Feed forward network.
  uint32_t pre_ffn_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(
      b.RmsNorm(residual_sum_id, weights.gamma[1], kHidden, pre_ffn_id));

  uint32_t ffn_q_id, ffn_zp_id, ffn_scale_id;
  RETURN_IF_ERROR(
      b.DynamicQuantize(pre_ffn_id, ffn_q_id, ffn_zp_id, ffn_scale_id));

  uint32_t gate_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(b.QuantizedDot(ffn_q_id, ffn_zp_id, ffn_scale_id,
                                 weights.w_gate, weights.s_gate, kHidden, kFfn,
                                 gate_id));
  uint32_t up_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(b.QuantizedDot(ffn_q_id, ffn_zp_id, ffn_scale_id,
                                 weights.w_up, weights.s_up, kHidden, kFfn,
                                 up_id));

  uint32_t activated_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(define_approx_gelu(subgraph, gate_id, activated_id));

  uint32_t gated_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply, activated_id,
                                    up_id, &gated_id, 0));

  uint32_t gated_q_id, gated_zp_id, gated_scale_id;
  RETURN_IF_ERROR(
      b.DynamicQuantize(gated_id, gated_q_id, gated_zp_id, gated_scale_id));

  uint32_t down_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(b.QuantizedDot(gated_q_id, gated_zp_id, gated_scale_id,
                                 weights.w_down, weights.s_down, kFfn, kHidden,
                                 down_id));

  uint32_t post_ffn_id = YNN_INVALID_VALUE_ID;
  RETURN_IF_ERROR(b.RmsNorm(down_id, weights.gamma[2], kHidden, post_ffn_id));

  // The layer output. In the model this is an external output of the graph,
  // which turns out to matter a lot for how the block is scheduled.
  if (!final_norm) {
    // Without the trailing normalization, the residual sum is the only output.
    return ynn_define_binary(subgraph, ynn_binary_add, residual_sum_id,
                             post_ffn_id, &norm_out_id, 0);
  }

  uint32_t block_out = YNN_INVALID_VALUE_ID;
  if (external_residual_output) {
    block_out = block_out_id;
    RETURN_IF_ERROR(ynn_define_tensor(subgraph, ynn_type_fp32, 2, nullptr,
                                      nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                      &block_out));
  }
  RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add, residual_sum_id,
                                    post_ffn_id, &block_out, 0));

  return b.RmsNorm(block_out, weights.gamma[3], kHidden, norm_out_id);
}

void BM_TransformerBlock(benchmark::State& state) {
  const size_t tokens = state.range(0);
  const bool static_shape = state.range(1);
  const int thread_count = state.range(2);
  const bool final_norm = state.range(3);
  const bool external_residual_output = final_norm && state.range(4);
  constexpr bool transpose_attn = true;

  static const BlockWeights weights;

  const uint32_t attn_id = 0;
  const uint32_t residual_id = 1;
  const uint32_t norm_out_id = 2;
  const uint32_t block_out_id = 3;

  SubgraphBuilder builder(external_residual_output ? 4 : 3);
  if (DefineBlock(builder.GetSubgraph(), weights, tokens, static_shape,
                  transpose_attn, external_residual_output, final_norm, attn_id,
                  residual_id, block_out_id,
                  norm_out_id) != ynn_status_success) {
    state.SkipWithError("failed to define block");
    return;
  }

  TestScheduler scheduler(thread_count - 1);
  Runtime runtime(builder.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  std::vector<float> attn(tokens * kAttnOut);
  std::vector<float> residual(tokens * kHidden);
  for (size_t i = 0; i < attn.size(); ++i) {
    attn[i] = 0.01f * static_cast<float>((i % 101) - 50);
  }
  for (size_t i = 0; i < residual.size(); ++i) {
    residual[i] = 0.01f * static_cast<float>((i % 61) - 30);
  }
  std::vector<float> norm_out(tokens * kHidden);
  std::vector<float> block_out(tokens * kHidden);

  if (transpose_attn) {
    runtime.ReshapeExternalTensor({kHeads, tokens, kHeadDim}, attn.data(),
                                  attn_id);
  } else {
    runtime.ReshapeExternalTensor({tokens, kAttnOut}, attn.data(), attn_id);
  }
  runtime.ReshapeExternalTensor({tokens, kHidden}, residual.data(),
                                residual_id);
  runtime.ReshapeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }
  runtime.SetupExternalTensor(norm_out.data(), norm_out_id);
  if (external_residual_output) {
    runtime.SetupExternalTensor(block_out.data(), block_out_id);
  }

  for (auto _ : state) {
    runtime.InvokeRuntime();
  }
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("failed to invoke runtime");
    return;
  }

  // Report a checksum so that mis-scheduled (e.g. racy) variants are visible,
  // and the MACs of the five matmuls in the block.
  double checksum = 0.0;
  for (float v : norm_out) checksum += v;
  state.counters["checksum"] = checksum;

  const size_t macs =
      tokens * (kAttnOut * kHidden + 3 * kHidden * kFfn + kFfn * kHidden);
  state.counters["MAC"] = benchmark::Counter(
      static_cast<double>(state.iterations() * macs), benchmark::Counter::kIsRate);
}

void Arguments(benchmark::Benchmark* b) {
  b->ArgNames(
      {"tokens", "static", "threads", "final_norm", "external_residual"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int tokens : {1024}) {
    for (int threads : {1, 4}) {
      for (bool static_shape : {false, true}) {
        // The graph as it appears in the model: the layer output is an
        // external output of the graph which the trailing normalization also
        // consumes.
        b->Args({tokens, static_shape, threads, /*final_norm=*/true,
                 /*external_residual=*/true});
        // Ablations. Making the layer output an internal value lets slinky
        // crop it, and dropping the trailing normalization leaves no loop for
        // the block to be fused into; both should run at the same speed.
        b->Args({tokens, static_shape, threads, /*final_norm=*/true,
                 /*external_residual=*/false});
        b->Args({tokens, static_shape, threads, /*final_norm=*/false,
                 /*external_residual=*/false});
      }
    }
  }
}

BENCHMARK(BM_TransformerBlock)
    ->Apply(Arguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynn
