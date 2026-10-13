// qwen.fused_gemm (R=2 capable build): SkinnyGemm, SharedExpertDown, ArgmaxRows.
// Every row of an M-row call (M <= 4) is computed with exactly the arithmetic and reduction order of the
// M == 1 call (the config never depends on M), so row r of an S=2 verify is bit-identical to the S=1 result.
// qwen.fused_gemm::SkinnyGemm - BF16 decode GEMV/GEMM (M <= 4) that writes each split slice directly.
//
//   inputs : x [..., K], W_0 [K, N_0], ..., W_{g-1} [K, N_{g-1}]   (all BF16, row-major weights)
//   outputs: one BF16 tensor [..., width_j] per entry of `widths`
//   attrs  : seg_outs  int[g]  number of outputs that slice W_i's columns (in order)
//            widths    int[]   output widths, grouped by weight (sum over weight i == N_i)
// Equivalent to MatMul(x, W_i) followed by Split(widths of W_i) for every weight i. Accumulation is
// FP32, the result is rounded once to BF16; the reduction order is fixed (deterministic).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "argmax_rows.cuh"
#include "skinny_gemm.cuh"

namespace {

using skinny::bf16;

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

// Optional node attribute `trigger` (int, default 0): 1 = launch_dependents at kernel start, 2 = after the K loop. It is set (by
// optimize_v12_hc.py) only on gemm nodes whose NEXT node in the stream is a qwen.fused_hc op, the only consumers that are known to
// call griddepcontrol.wait before touching the data. SKINNY_TRIGGER_FORCE=<n> overrides a non-zero attribute (experiments).
int TriggerAttr(const OrtKernelInfo* info) {
  int v = 0;
  try {
    v = static_cast<int>(Ort::ConstKernelInfo(info).GetAttribute<int64_t>("trigger"));
  } catch (const std::exception&) {
  }
  const char* e = std::getenv("SKINNY_TRIGGER_FORCE");
  return (v != 0 && e != nullptr) ? std::atoi(e) : v;
}

// Tuned on H200 (see REPORT.md): {tx, cluster, threads, unroll} per (K, total N, segments).
const bf16* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const bf16*>(c.GetInput(i).GetTensorData<Ort::BFloat16_t>());
}
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

skinny::Config Choose(int m, int K, int nseg, const int* n) {
  int total = 0;
  for (int s = 0; s < nseg; ++s) total += n[s];
  struct Entry {
    int K, total, nseg;
    skinny::Config cfg;
  };
  static const Entry table[] = {
      {2560, 16480, 1, {16, 1, 1024, 8}},
      {2560, 13952, 1, {8, 1, 512, 8}},
      {2560, 12800, 1, {8, 1, 512, 8}},
      {2560, 1792, 1, {8, 8, 512, 4}},
      {6144, 2560, 1, {8, 8, 256, 8}},
      {640, 2560, 1, {4, 1, 512, 8}},
      {10240, 320, 1, {4, 8, 1024, 8}},
      {320, 10240, 1, {16, 1, 1024, 8}},
      {10240, 324, 2, {4, 8, 1024, 8}},
      {2560, 248320, 1, {16, 1, 256, 8}},
  };
  // The config (hence the reduction order) is deliberately independent of the number of rows.
  (void)m;
  for (const auto& e : table) {
    if (e.K == K && e.total == total && e.nseg == nseg) return e.cfg;
  }
  // Generic fallback: ~256 columns per CTA, K split over a cluster so that >= ~256 CTAs run.
  skinny::Config cfg;
  cfg.tx = 8;
  cfg.threads = 256;
  cfg.unroll = 4;
  const long tiles = (total + 63) / 64;
  cfg.cluster = tiles >= 256 ? 1 : tiles >= 128 ? 2 : tiles >= 64 ? 4 : 8;
  return cfg;
}

struct SkinnyGemmKernelState {
  const OrtApi& api;
  std::vector<int> seg_outs;
  std::vector<int> widths;
  int trigger = 0;

  SkinnyGemmKernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    trigger = TriggerAttr(info);
    for (auto v : k.GetAttributes<int64_t>("seg_outs")) seg_outs.push_back(static_cast<int>(v));
    for (auto v : k.GetAttributes<int64_t>("widths")) widths.push_back(static_cast<int>(v));
    int total = 0;
    for (int v : seg_outs) total += v;
    if (seg_outs.empty() || seg_outs.size() > skinny::kMaxSeg || total != static_cast<int>(widths.size()) ||
        total > skinny::kMaxOut) {
      throw std::invalid_argument("SkinnyGemm: bad seg_outs/widths");
    }
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const int nseg = static_cast<int>(seg_outs.size());
      if (c.GetInputCount() != static_cast<size_t>(nseg) + 1) throw std::invalid_argument("SkinnyGemm: input count");
      auto x = c.GetInput(0);
      auto xshape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int K = static_cast<int>(xshape.back());
      int64_t m64 = 1;
      for (size_t i = 0; i + 1 < xshape.size(); ++i) m64 *= xshape[i];
      if (m64 < 1 || m64 > 8) throw std::invalid_argument("SkinnyGemm: M must be in [1, 8]");
      const int M = static_cast<int>(m64);

      skinny::Params p = {};
      p.trigger = trigger;
      p.x = reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>());
      p.K = K;
      p.nseg = nseg;
      int o = 0;
      for (int s = 0; s < nseg; ++s) {
        auto w = c.GetInput(1 + s);
        auto wshape = w.GetTensorTypeAndShapeInfo().GetShape();
        if (wshape.size() != 2 || wshape[0] != K) throw std::invalid_argument("SkinnyGemm: weight shape");
        p.w[s] = reinterpret_cast<const bf16*>(w.GetTensorData<Ort::BFloat16_t>());
        p.n[s] = static_cast<int>(wshape[1]);
        if (p.n[s] % 4 != 0) throw std::invalid_argument("SkinnyGemm: N must be a multiple of 4");
        if (K % 2 != 0) throw std::invalid_argument("SkinnyGemm: K must be even");
        p.out_begin[s] = o;
        int begin = 0;
        for (int i = 0; i < seg_outs[s]; ++i, ++o) {
          std::vector<int64_t> shape(xshape.begin(), xshape.end() - 1);
          shape.push_back(widths[o]);
          auto out = c.GetOutput(o, shape);
          p.out[o] = reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>());
          p.col_begin[o] = begin;
          p.width[o] = widths[o];
          begin += widths[o];
        }
        if (begin != p.n[s]) throw std::invalid_argument("SkinnyGemm: widths do not add up to N");
      }
      p.out_begin[nseg] = o;
      const skinny::Config cfg = Choose(M, K, nseg, p.n);
      cudaError_t err = skinny::Launch(M, p, cfg, Stream(api, raw));
      if (err == cudaSuccess) err = cudaGetLastError();
      if (err != cudaSuccess) throw std::runtime_error(std::string("SkinnyGemm: ") + cudaGetErrorString(err));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// y = routed + bf16(bf16(swish_mul(g, u) @ W) * bf16(sigmoid(bf16(dot(gx, gw))))) per row, M in [1, 4].
struct SharedExpertDownKernelState {
  const OrtApi& api;
  int trigger = 0;
  SharedExpertDownKernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a), trigger(TriggerAttr(info)) {}

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      if (c.GetInputCount() != 6) throw std::invalid_argument("SharedExpertDown: needs 6 inputs");
      auto g = c.GetInput(0);
      auto w = c.GetInput(2);
      auto wshape = w.GetTensorTypeAndShapeInfo().GetShape();
      if (wshape.size() != 2) throw std::invalid_argument("SharedExpertDown: weight rank");
      const int64_t K = wshape[0], N = wshape[1];
      const int64_t count_g = Count(g), count_u = Count(c.GetInput(1)), count_gx = Count(c.GetInput(3));
      const int64_t gk = Count(c.GetInput(4));
      const int64_t M = K > 0 ? count_g / K : 0;
      if (M < 1 || M > 4 || count_g != M * K || count_u != M * K || Count(c.GetInput(5)) != M * N ||
          K > skinny::kFuseMaxK || N % 8 != 0 || gk % 2 != 0 || count_gx != M * gk ||
          c.GetInput(4).GetTensorTypeAndShapeInfo().GetShape().back() != 1) {
        throw std::invalid_argument("SharedExpertDown: shape mismatch (M must be in [1, 4])");
      }
      auto out = c.GetOutput(0, c.GetInput(5).GetTensorTypeAndShapeInfo().GetShape());
      skinny::Params p = {};
      p.trigger = trigger;
      p.x = In(c, 0);
      p.x2 = In(c, 1);
      p.K = static_cast<int>(wshape[0]);
      p.nseg = 1;
      p.w[0] = In(c, 2);
      p.n[0] = static_cast<int>(wshape[1]);
      p.out_begin[0] = 0;
      p.out_begin[1] = 1;
      p.out[0] = reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>());
      p.col_begin[0] = 0;
      p.width[0] = p.n[0];
      p.gx = In(c, 3);
      p.gw = In(c, 4);
      p.routed = In(c, 5);
      p.gk = static_cast<int>(gk);
      skinny::Config cfg;
      cfg.tx = 4;
      cfg.cluster = 1;
      cfg.threads = 512;
      cfg.unroll = 8;
      cudaError_t err = skinny::LaunchFused(static_cast<int>(M), p, cfg, Stream(api, raw));
      if (err == cudaSuccess) err = cudaGetLastError();
      if (err != cudaSuccess) throw std::runtime_error(std::string("SharedExpertDown: ") + cudaGetErrorString(err));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// idx[..., ] = argmax over the last axis of logits[..., V] (BF16/FP16/FP32), numpy semantics (lowest index on ties,
// first NaN wins). Optional second output: the maximum value as FP32.
struct ArgmaxRowsKernelState {
  const OrtApi& api;
  explicit ArgmaxRowsKernelState(const OrtApi& a) : api(a) {}

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      if (c.GetInputCount() != 1 || c.GetOutputCount() < 1 || c.GetOutputCount() > 2) {
        throw std::invalid_argument("ArgmaxRows: 1 input, 1 or 2 outputs");
      }
      auto x = c.GetInput(0);
      auto info = x.GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      if (shape.empty() || shape.back() < 1) throw std::invalid_argument("ArgmaxRows: bad shape");
      skinny::LogitType type;
      const void* data = nullptr;
      switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
          type = skinny::LogitType::kBF16;
          data = x.GetTensorData<Ort::BFloat16_t>();
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          type = skinny::LogitType::kFP16;
          data = x.GetTensorData<Ort::Float16_t>();
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          type = skinny::LogitType::kFP32;
          data = x.GetTensorData<float>();
          break;
        default: throw std::invalid_argument("ArgmaxRows: logits must be BF16/FP16/FP32");
      }
      const int64_t V = shape.back();
      std::vector<int64_t> oshape(shape.begin(), shape.end() - 1);
      int64_t rows = 1;
      for (auto d : oshape) rows *= d;
      auto idx = c.GetOutput(0, oshape);
      float* val = nullptr;
      if (c.GetOutputCount() == 2) val = c.GetOutput(1, oshape).GetTensorMutableData<float>();
      if (rows == 0) return nullptr;
      cudaError_t err = skinny::LaunchArgmaxRows(type, data, rows, V, idx.GetTensorMutableData<int64_t>(), val,
                                                 Stream(api, raw));
      if (err == cudaSuccess) err = cudaGetLastError();
      if (err != cudaSuccess) throw std::runtime_error(std::string("ArgmaxRows: ") + cudaGetErrorString(err));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct ArgmaxRowsOp : Ort::CustomOpBase<ArgmaxRowsOp, ArgmaxRowsKernelState, true> {
  const char* GetName() const { return "ArgmaxRows"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_OPTIONAL;
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo*, void** kernel) const noexcept {
    *kernel = new ArgmaxRowsKernelState(api);
    return nullptr;
  }
};

struct SharedExpertDownOp : Ort::CustomOpBase<SharedExpertDownOp, SharedExpertDownKernelState, true> {
  const char* GetName() const { return "SharedExpertDown"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 6; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new SharedExpertDownKernelState(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

struct SkinnyGemmOp : Ort::CustomOpBase<SkinnyGemmOp, SkinnyGemmKernelState, true> {
  const char* GetName() const { return "SkinnyGemm"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_VARIADIC;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_VARIADIC; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new SkinnyGemmKernelState(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static SkinnyGemmOp skinny_gemm;
    static SharedExpertDownOp shared_down;
    static ArgmaxRowsOp argmax_rows;
    static Ort::CustomOpDomain domain("qwen.fused_gemm");
    static std::once_flag registered;
    std::call_once(registered, [&]() { domain.Add(&skinny_gemm);
      domain.Add(&shared_down);
      domain.Add(&argmax_rows);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
