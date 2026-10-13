// Accurate BF16 sigmoid shared by the *_r5 libraries (qwen.fused, qwen.fused_attn, qwen.fused_hc, qwen.fused_gemm).
// Bit-identical to onnxruntime's CUDA EP OP_Sigmoid<BFloat16> after the fix on branch qwen38-prefill-fix (commit 453ac97a68):
// evaluate in fp32 (exp(-|a|), 1/(1+e) for a > 0 and e/(1+e) otherwise, which keeps the relative precision of small results) and
// round to BF16 once.  The old operator rounded after every operation (1 - 1/(1+exp(a)) in BF16): up to 100 % relative error.
// The formula must stay in sync with the provider: tests/sigmoid_exhaustive.py compares it against the ORT node for all 65536 bit patterns.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qwen_acc {

__device__ __forceinline__ float SigmoidF(float a) {
  const float e = expf(-fabsf(a));
  return a > 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// the BF16 value of the ORT Sigmoid node, as a float
__device__ __forceinline__ float SigmoidBf16(float a) { return __bfloat162float(__float2bfloat16(SigmoidF(a))); }

// ORT `Swish` is the function body Mul(x, Sigmoid(x)): BF16 sigmoid first, then one BF16 rounding of the product
__device__ __forceinline__ float SwishBf16(float x) { return __bfloat162float(__float2bfloat16(x * SigmoidBf16(x))); }

}  // namespace qwen_acc
