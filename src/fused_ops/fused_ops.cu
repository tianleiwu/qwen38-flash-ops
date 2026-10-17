// Fused BF16 pointwise decode ops for the Qwen3.8 S=1 capture graph.
// Every op rounds to BF16 at the same points as the unfused ONNX chain (ORT Sigmoid = accurate BF16 sigmoid of provider commit 453ac97a68; Swish = Mul(x, Sigmoid(x))).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include "common/sigmoid_acc.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float R(float v) {
  return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ float Sigmoid(float v) {
  return 1.0f / (1.0f + expf(-v));
}

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

const bf16* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const bf16*>(c.GetInput(i).GetTensorData<Ort::BFloat16_t>());
}
bf16* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<bf16*>(v.GetTensorMutableData<Ort::BFloat16_t>());
}
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

// y = bf16(bf16(sigmoid(bf16(x / div))) * mul)
__global__ void DivSigmoidMulKernel(const bf16* x, bf16* y, int64_t n, float div, float mul) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float t = R(__bfloat162float(x[i]) / div);
  t = R(Sigmoid(t));
  y[i] = __float2bfloat16(t * mul);
}

// y = bf16(bf16(silu(g)) * u)
__global__ void SwishMulKernel(const bf16* g, const bf16* u, bf16* y, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  float v = __bfloat162float(g[i]);
  float s = R(v * Sigmoid(v));
  y[i] = __float2bfloat16(s * __bfloat162float(u[i]));
}

// logit = bf16(dot(x[r], w)); y[r] = bf16(routed[r] + bf16(shared[r] * bf16(sigmoid(logit))))
__global__ void SharedGateMulAddKernel(
    const bf16* x, const bf16* w, const bf16* shared, const bf16* routed, bf16* y, int k, int cols) {
  const int row = blockIdx.y;
  const bf16* xr = x + static_cast<int64_t>(row) * k;
  const bf16* sr = shared + static_cast<int64_t>(row) * cols;
  const bf16* rr = routed + static_cast<int64_t>(row) * cols;
  bf16* yr = y + static_cast<int64_t>(row) * cols;
  __shared__ float partial[32];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    sum += __bfloat162float(xr[i]) * __bfloat162float(w[i]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    if (threadIdx.x == 0) partial[0] = v;
  }
  __syncthreads();
  const float gate = R(Sigmoid(R(partial[0])));
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += gridDim.x * blockDim.x) {
    float s = R(__bfloat162float(sr[i]) * gate);
    yr[i] = __float2bfloat16(__bfloat162float(rr[i]) + s);
  }
}

struct DivSigmoidMulKernelState {
  const OrtApi& api;
  float div, mul;
  DivSigmoidMulKernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    div = k.GetAttribute<float>("div");
    mul = k.GetAttribute<float>("mul");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      auto out = c.GetOutput(0, shape);
      int64_t n = Count(x);
      DivSigmoidMulKernel<<<(n + 255) / 256, 256, 0, Stream(api, raw)>>>(In(c, 0), Out(out), n, div, mul);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct SwishMulKernelState {
  const OrtApi& api;
  SwishMulKernelState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto g = c.GetInput(0);
      auto shape = g.GetTensorTypeAndShapeInfo().GetShape();
      if (Count(c.GetInput(1)) != Count(g)) throw std::invalid_argument("SwishMul: shape mismatch");
      auto out = c.GetOutput(0, shape);
      int64_t n = Count(g);
      SwishMulKernel<<<(n + 255) / 256, 256, 0, Stream(api, raw)>>>(In(c, 0), In(c, 1), Out(out), n);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct SharedGateMulAddKernelState {
  const OrtApi& api;
  SharedGateMulAddKernelState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto shared = c.GetInput(2);
      auto shape = shared.GetTensorTypeAndShapeInfo().GetShape();
      int64_t n = Count(shared);
      int64_t k = Count(c.GetInput(1));
      int64_t x_cnt = Count(c.GetInput(0));
      int64_t rows = k > 0 ? x_cnt / k : 0;
      int64_t cols = rows > 0 ? n / rows : 0;
      if (rows < 1 || x_cnt != rows * k || cols * rows != n || Count(c.GetInput(3)) != n) {
        throw std::invalid_argument("SharedGateMulAdd: shape mismatch");
      }
      auto out = c.GetOutput(0, shape);
      int blocks_x = rows == 1 ? static_cast<int>(std::min<int64_t>((cols + 255) / 256, 16)) : 1;
      dim3 grid(blocks_x, static_cast<unsigned>(rows));
      SharedGateMulAddKernel<<<grid, 256, 0, Stream(api, raw)>>>(
          In(c, 0), In(c, 1), In(c, 2), In(c, 3), Out(out), static_cast<int>(k), static_cast<int>(cols));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

template <typename Derived, typename State, size_t kInputs>
struct BF16Op : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return kInputs; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

struct DivSigmoidMulOp : BF16Op<DivSigmoidMulOp, DivSigmoidMulKernelState, 1> {
  const char* GetName() const { return "DivSigmoidMul"; }
};
struct SwishMulOp : BF16Op<SwishMulOp, SwishMulKernelState, 2> {
  const char* GetName() const { return "SwishMul"; }
};
struct SharedGateMulAddOp : BF16Op<SharedGateMulAddOp, SharedGateMulAddKernelState, 4> {
  const char* GetName() const { return "SharedGateMulAdd"; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static DivSigmoidMulOp div_sigmoid_mul;
    static SwishMulOp swish_mul;
    static SharedGateMulAddOp shared_gate;
    static Ort::CustomOpDomain domain("qwen.fused");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&div_sigmoid_mul);
      domain.Add(&swish_mul);
      domain.Add(&shared_gate);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
