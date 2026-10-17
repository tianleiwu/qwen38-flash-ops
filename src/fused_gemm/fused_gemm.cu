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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "argmax_rows.cuh"
#include "skinny_gemm.cuh"

namespace {

struct QMoESharedExpertBridge {
  static constexpr uint64_t kMagic = 0x514d4f4553484558ULL;
  static constexpr int kMaxRegisteredNames = 256;
  static constexpr int kMaxNameLen = 128;
  uint64_t magic = kMagic;
  std::atomic<int> lock{0};
  std::atomic<int> num_registered{0};
  char registered_routed_names[kMaxRegisteredNames][kMaxNameLen]{};
  bool pending = false;
  cudaStream_t stream = nullptr;
  const void* routed_output_ptr = nullptr;
  int tile_rows = 0;
  int hidden_size = 0;
  int (*launch_deferred)(void* ctx, cudaStream_t stream, void* shared_out_ptr, int trigger) = nullptr;
  void* ctx = nullptr;

  void RegisterRoutedName(const char* name) noexcept {
    if (!name || !*name) return;
    while (lock.exchange(1, std::memory_order_acquire) != 0) {
    }
    const int n = num_registered.load(std::memory_order_relaxed);
    bool exists = false;
    for (int i = 0; i < n; ++i) {
      if (std::strncmp(registered_routed_names[i], name, kMaxNameLen - 1) == 0) {
        exists = true;
        break;
      }
    }
    if (!exists && n < kMaxRegisteredNames) {
      std::strncpy(registered_routed_names[n], name, kMaxNameLen - 1);
      registered_routed_names[n][kMaxNameLen - 1] = '\0';
      num_registered.store(n + 1, std::memory_order_release);
    }
    lock.store(0, std::memory_order_release);
  }
};

inline QMoESharedExpertBridge* GetQMoESharedExpertBridge() noexcept {
  static QMoESharedExpertBridge* const instance = []() -> QMoESharedExpertBridge* {
    constexpr const char* kEnvVar = "ORT_QMOE_SHARED_EXPERT_BRIDGE_PTR";
    if (const char* env = std::getenv(kEnvVar); env && *env) {
      unsigned long long raw_addr = 0;
      if (std::sscanf(env, "%llx", &raw_addr) == 1 && raw_addr != 0) {
        auto* p = reinterpret_cast<QMoESharedExpertBridge*>(static_cast<uintptr_t>(raw_addr));
        if (p->magic == QMoESharedExpertBridge::kMagic) return p;
      }
    }
    auto* p = new QMoESharedExpertBridge();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
#if defined(_WIN32)
    _putenv_s(kEnvVar, buf);
#else
    setenv(kEnvVar, buf, 1);
#endif
    return p;
  }();
  return instance;
}

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

skinny::Config Choose(int m, int K, int nseg, const int* n, int w_mode = 0) {
  int total = 0;
  for (int s = 0; s < nseg; ++s) total += n[s];
  struct Entry {
    int K, total, nseg;
    skinny::Config cfg;
  };
  static const Entry table_bf16[] = {
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
  static const Entry table_fp8[] = {
      {2560, 16480, 1, {16, 1, 1024, 8}},
      {2560, 13952, 1, {16, 1, 1024, 8}},
      {2560, 12800, 1, {16, 1, 1024, 8}},
      {2560, 1792, 1, {8, 8, 512, 8}},
      {6144, 2560, 1, {8, 8, 256, 8}},
      {2560, 2560, 1, {8, 8, 256, 4}},
      {640, 2560, 1, {4, 1, 512, 8}},
      {10240, 320, 1, {4, 8, 512, 4}},
      {320, 10240, 1, {16, 1, 1024, 8}},
      {10240, 324, 2, {4, 8, 1024, 8}},
      {2560, 248320, 1, {32, 2, 256, 8}},
  };
  // The config (hence the reduction order) is deliberately independent of the number of rows.
  (void)m;
  if (w_mode == 1) {
    for (const auto& e : table_fp8) {
      if (e.K == K && e.total == total && e.nseg == nseg) return e.cfg;
    }
  }
  for (const auto& e : table_bf16) {
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
  float* d_w_scale = nullptr;
  size_t w_scale_len = 0;

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
    std::vector<float> scale_host;
    try {
      scale_host = k.GetAttributes<float>("w_scale");
    } catch (const std::exception&) {
    }
    if (!scale_host.empty()) {
      w_scale_len = scale_host.size();
      if (cudaMalloc(&d_w_scale, w_scale_len * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_w_scale, scale_host.data(), w_scale_len * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w_scale);
        d_w_scale = nullptr;
        throw std::runtime_error("SkinnyGemm: failed to upload w_scale");
      }
    }
  }

  ~SkinnyGemmKernelState() {
    cudaFree(d_w_scale);
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
      if (m64 < 1 || m64 > 32) throw std::invalid_argument("SkinnyGemm: M must be in [1, 32]");
      const int M = static_cast<int>(m64);

      skinny::Params p = {};
      p.trigger = trigger;
      p.x = reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>());
      p.K = K;
      p.nseg = nseg;
      int o = 0;
      size_t scale_off = 0;
      for (int s = 0; s < nseg; ++s) {
        auto w = c.GetInput(1 + s);
        auto winfo = w.GetTensorTypeAndShapeInfo();
        auto wshape = winfo.GetShape();
        const auto wtype = winfo.GetElementType();
        if (wshape.size() != 2 || wshape[0] != K) throw std::invalid_argument("SkinnyGemm: weight shape");
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
        p.n[s] = begin;
        if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
          if (wshape[1] != begin) throw std::invalid_argument("SkinnyGemm: widths do not add up to N");
          p.w[s] = reinterpret_cast<const bf16*>(w.GetTensorData<Ort::BFloat16_t>());
          p.w_mode[s] = 0;
        } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
          if (wshape[1] != begin) throw std::invalid_argument("SkinnyGemm: FP8 widths do not add up to N");
          p.w_q[s] = static_cast<const uint8_t*>(w.GetTensorRawData());
          p.w_mode[s] = 1;
          if (d_w_scale != nullptr && scale_off + begin <= w_scale_len) {
            p.w_scale[s] = d_w_scale + scale_off;
            scale_off += begin;
          }
        } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
          if (begin % 16 != 0 || wshape[1] != (begin * 9) / 16)
            throw std::invalid_argument("SkinnyGemm: NVFP4 packed width must equal N * 9 / 16");
          p.w_q[s] = static_cast<const uint8_t*>(w.GetTensorRawData());
          p.w_mode[s] = 2;
          if (d_w_scale != nullptr && scale_off + begin <= w_scale_len) {
            p.w_scale[s] = d_w_scale + scale_off;
            scale_off += begin;
          }
        } else {
          throw std::invalid_argument("SkinnyGemm: unsupported weight element type");
        }
        if (p.n[s] % 4 != 0) throw std::invalid_argument("SkinnyGemm: N must be a multiple of 4");
        if (K % 2 != 0) throw std::invalid_argument("SkinnyGemm: K must be even");
      }
      p.out_begin[nseg] = o;
      const skinny::Config cfg = Choose(M, K, nseg, p.n, p.w_mode[0]);
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
  float* d_w_scale = nullptr;
  size_t w_scale_len = 0;

  SharedExpertDownKernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a), trigger(TriggerAttr(info)) {
    Ort::ConstKernelInfo k(info);
    try {
      if (k.GetInputCount() >= 6) {
        const std::string routed_name = k.GetInputName(5);
        GetQMoESharedExpertBridge()->RegisterRoutedName(routed_name.c_str());
      }
    } catch (const std::exception&) {
    }
    std::vector<float> scale_host;
    try {
      scale_host = k.GetAttributes<float>("w_scale");
    } catch (const std::exception&) {
    }
    if (!scale_host.empty()) {
      w_scale_len = scale_host.size();
      if (cudaMalloc(&d_w_scale, w_scale_len * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_w_scale, scale_host.data(), w_scale_len * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w_scale);
        d_w_scale = nullptr;
        throw std::runtime_error("SharedExpertDown: failed to upload w_scale");
      }
    }
  }

  ~SharedExpertDownKernelState() {
    cudaFree(d_w_scale);
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      if (c.GetInputCount() != 6) throw std::invalid_argument("SharedExpertDown: needs 6 inputs");
      auto g = c.GetInput(0);
      auto w = c.GetInput(2);
      auto winfo = w.GetTensorTypeAndShapeInfo();
      auto wshape = winfo.GetShape();
      const auto wtype = winfo.GetElementType();
      if (wshape.size() != 2) throw std::invalid_argument("SharedExpertDown: weight rank");
      const int64_t K = wshape[0];
      const int64_t N = c.GetInput(5).GetTensorTypeAndShapeInfo().GetShape().back();
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
      p.K = static_cast<int>(K);
      p.nseg = 1;
      p.n[0] = static_cast<int>(N);
      if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        if (wshape[1] != N) throw std::invalid_argument("SharedExpertDown: BF16 weight N mismatch");
        p.w[0] = In(c, 2);
        p.w_mode[0] = 0;
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
        if (wshape[1] != N) throw std::invalid_argument("SharedExpertDown: FP8 weight N mismatch");
        p.w_q[0] = static_cast<const uint8_t*>(w.GetTensorRawData());
        p.w_mode[0] = 1;
        if (d_w_scale != nullptr && w_scale_len >= static_cast<size_t>(N)) p.w_scale[0] = d_w_scale;
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        if (N % 16 != 0 || wshape[1] != (N * 9) / 16)
          throw std::invalid_argument("SharedExpertDown: NVFP4 weight N mismatch");
        p.w_q[0] = static_cast<const uint8_t*>(w.GetTensorRawData());
        p.w_mode[0] = 2;
        if (d_w_scale != nullptr && w_scale_len >= static_cast<size_t>(N)) p.w_scale[0] = d_w_scale;
      } else {
        throw std::invalid_argument("SharedExpertDown: unsupported weight element type");
      }
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
      if (p.w_mode[0] == 1) {
        cfg.tx = 8;
        cfg.cluster = 1;
        cfg.threads = 512;
      } else {
        cfg.tx = 4;
        cfg.cluster = 1;
        cfg.threads = 512;
      }
      cfg.unroll = 8;
      cudaStream_t stream = Stream(api, raw);
      auto* bridge = GetQMoESharedExpertBridge();
      if (bridge && bridge->pending && bridge->stream == stream && bridge->routed_output_ptr == p.routed &&
          bridge->launch_deferred != nullptr) {
        bridge->pending = false;
        skinny::Params p_proj = p;
        p_proj.routed = nullptr;
        p_proj.trigger = 0;
        cudaError_t err = skinny::LaunchFused(static_cast<int>(M), p_proj, cfg, stream);
        if (err == cudaSuccess) err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(std::string("SharedExpertDown(proj): ") + cudaGetErrorString(err));
        if (bridge->launch_deferred(bridge->ctx, stream, p.out[0], trigger) != 0) {
          throw std::runtime_error("SharedExpertDown: deferred QMoE merge+finalize+combine failed");
        }
        return nullptr;
      }
      cudaError_t err = skinny::LaunchFused(static_cast<int>(M), p, cfg, stream);
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
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
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
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_VARIADIC;
  }
  bool GetVariadicInputHomogeneity() const { return false; }
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

struct DequantOutParams {
  int num_outs;
  bf16* out[skinny::kMaxOut];
  int col_begin[skinny::kMaxOut];
  int width[skinny::kMaxOut];
};

__device__ __forceinline__ uint4 DeqFp8x8ToBf16U4(uint2 raw, const float* w_scale, int col) {
  const float4 s0 = w_scale ? __ldg(reinterpret_cast<const float4*>(w_scale + col)) : make_float4(1.f, 1.f, 1.f, 1.f);
  const float4 s1 = w_scale ? __ldg(reinterpret_cast<const float4*>(w_scale + col + 4)) : make_float4(1.f, 1.f, 1.f, 1.f);
  const __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw.x & 0xffffu), __NV_E4M3);
  const __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw.x >> 16), __NV_E4M3);
  const __half2_raw hr2 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw.y & 0xffffu), __NV_E4M3);
  const __half2_raw hr3 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(raw.y >> 16), __NV_E4M3);
  const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&hr0));
  const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hr1));
  const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&hr2));
  const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&hr3));
  const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0.x * s0.x, f0.y * s0.y);
  const __nv_bfloat162 b1 = __floats2bfloat162_rn(f1.x * s0.z, f1.y * s0.w);
  const __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x * s1.x, f2.y * s1.y);
  const __nv_bfloat162 b3 = __floats2bfloat162_rn(f3.x * s1.z, f3.y * s1.w);
  return make_uint4(*reinterpret_cast<const uint32_t*>(&b0), *reinterpret_cast<const uint32_t*>(&b1),
                    *reinterpret_cast<const uint32_t*>(&b2), *reinterpret_cast<const uint32_t*>(&b3));
}

template <int WMODE, bool MULTI_OUT>
__global__ void __launch_bounds__(256)
    DequantWeightBf16Kernel(const uint8_t* wq, const float* w_scale, DequantOutParams op, int K, int N, int row_stride, int col_offset) {
  const int chunks_per_row = N / 8;
  const int64_t total_chunks = static_cast<int64_t>(K) * chunks_per_row;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_chunks;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(idx / chunks_per_row);
    const int col = static_cast<int>(idx % chunks_per_row) * 8;
    const int src_col = col_offset + col;
    uint4 u4;
    if constexpr (WMODE == 0) {
      u4 = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(wq) + static_cast<size_t>(r) * row_stride + src_col));
    } else if constexpr (WMODE == 1) {
      const uint2 raw = __ldg(reinterpret_cast<const uint2*>(wq + static_cast<size_t>(r) * row_stride + src_col));
      u4 = DeqFp8x8ToBf16U4(raw, w_scale, col);
    } else {
      const size_t fp4_row_bytes = static_cast<size_t>(row_stride) * 9 / 16;
      const uint8_t* row_p = wq + static_cast<size_t>(r) * fp4_row_bytes;
      const uint32_t pack8 = __ldg(reinterpret_cast<const uint32_t*>(row_p + (src_col >> 1)));
      const uint8_t s_fp8 = row_p[(row_stride >> 1) + (src_col >> 4)];
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s_fp8), __NV_E4M3);
      const float s_half = __half2float(*reinterpret_cast<const __half*>(&hr)) * 0.5f;
      const float4 s0 = w_scale ? __ldg(reinterpret_cast<const float4*>(w_scale + col)) : make_float4(1.f, 1.f, 1.f, 1.f);
      const float4 s1 = w_scale ? __ldg(reinterpret_cast<const float4*>(w_scale + col + 4)) : make_float4(1.f, 1.f, 1.f, 1.f);
      const float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      uint32_t words_out[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float f0 = skinny::E2M1ScaledToFloat((pack8 >> (8 * i)) & 0xfu, s_half) * sc[2 * i];
        const float f1 = skinny::E2M1ScaledToFloat((pack8 >> (8 * i + 4)) & 0xfu, s_half) * sc[2 * i + 1];
        const __nv_bfloat162 b = __floats2bfloat162_rn(f0, f1);
        words_out[i] = *reinterpret_cast<const uint32_t*>(&b);
      }
      u4 = make_uint4(words_out[0], words_out[1], words_out[2], words_out[3]);
    }
    if constexpr (!MULTI_OUT) {
      *reinterpret_cast<uint4*>(op.out[0] + static_cast<size_t>(r) * N + col) = u4;
    } else {
      int j = 0;
#pragma unroll
      for (int i = 1; i < skinny::kMaxOut; ++i) {
        if (i < op.num_outs && col >= op.col_begin[i]) j = i;
      }
      *reinterpret_cast<uint4*>(op.out[j] + static_cast<size_t>(r) * op.width[j] + (col - op.col_begin[j])) = u4;
    }
  }
}

struct DequantWeightBf16KernelState {
  const OrtApi& api;
  int64_t n_cols = 0;
  int64_t col_offset = 0;
  std::vector<int> widths;
  float* d_w_scale = nullptr;
  size_t w_scale_len = 0;

  DequantWeightBf16KernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    n_cols = k.GetAttribute<int64_t>("n_cols");
    try {
      col_offset = k.GetAttribute<int64_t>("col_offset");
    } catch (const std::exception&) {
      col_offset = 0;
    }
    try {
      for (auto v : k.GetAttributes<int64_t>("widths")) widths.push_back(static_cast<int>(v));
    } catch (const std::exception&) {
    }
    if (widths.empty()) widths.push_back(static_cast<int>(n_cols));
    int sum_w = 0;
    for (int v : widths) {
      if (v <= 0 || v % 8 != 0) throw std::invalid_argument("DequantWeightBf16: each width must be a positive multiple of 8");
      sum_w += v;
    }
    if (sum_w != n_cols || widths.size() > skinny::kMaxOut || col_offset < 0 || col_offset % 8 != 0)
      throw std::invalid_argument("DequantWeightBf16: invalid widths or col_offset");
    std::vector<float> scale_host;
    try {
      scale_host = k.GetAttributes<float>("w_scale");
    } catch (const std::exception&) {
    }
    if (!scale_host.empty()) {
      w_scale_len = scale_host.size();
      if (cudaMalloc(&d_w_scale, w_scale_len * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_w_scale, scale_host.data(), w_scale_len * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w_scale);
        d_w_scale = nullptr;
        throw std::runtime_error("DequantWeightBf16: failed to upload w_scale");
      }
    }
  }

  ~DequantWeightBf16KernelState() {
    cudaFree(d_w_scale);
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto w = c.GetInput(0);
      auto winfo = w.GetTensorTypeAndShapeInfo();
      auto wshape = winfo.GetShape();
      const auto wtype = winfo.GetElementType();
      if (wshape.size() != 2 || n_cols <= 0 || n_cols % 8 != 0)
        throw std::invalid_argument("DequantWeightBf16: invalid shape");
      const int64_t K = wshape[0];
      const int64_t N = n_cols;
      int w_mode = 0;
      int row_stride = static_cast<int>(N);
      if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 && wshape[1] >= col_offset + N) {
        w_mode = 0;
        row_stride = static_cast<int>(wshape[1]);
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN && wshape[1] >= col_offset + N) {
        w_mode = 1;
        row_stride = static_cast<int>(wshape[1]);
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 && N % 16 == 0 && col_offset % 16 == 0 &&
                 wshape[1] * 16 >= (col_offset + N) * 9 && (wshape[1] * 16) % 9 == 0) {
        w_mode = 2;
        row_stride = static_cast<int>((wshape[1] * 16) / 9);
      } else {
        throw std::invalid_argument("DequantWeightBf16: unsupported input type/shape");
      }
      DequantOutParams op = {};
      op.num_outs = static_cast<int>(widths.size());
      int col_off = 0;
      for (int i = 0; i < op.num_outs; ++i) {
        auto out = c.GetOutput(i, {K, widths[i]});
        op.out[i] = reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>());
        op.col_begin[i] = col_off;
        op.width[i] = widths[i];
        col_off += widths[i];
      }
      const int64_t total_chunks = K * (N / 8);
      const int threads = 256;
      const int blocks = static_cast<int>(std::min<int64_t>(8192, (total_chunks + threads - 1) / threads));
      const uint8_t* wq_ptr = static_cast<const uint8_t*>(w.GetTensorRawData());
      cudaStream_t stream = Stream(api, raw);
      const int k_i = static_cast<int>(K), n_i = static_cast<int>(N), co_i = static_cast<int>(col_offset);
      if (op.num_outs == 1) {
        if (w_mode == 0)
          DequantWeightBf16Kernel<0, false><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
        else if (w_mode == 1)
          DequantWeightBf16Kernel<1, false><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
        else
          DequantWeightBf16Kernel<2, false><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
      } else {
        if (w_mode == 0)
          DequantWeightBf16Kernel<0, true><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
        else if (w_mode == 1)
          DequantWeightBf16Kernel<1, true><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
        else
          DequantWeightBf16Kernel<2, true><<<blocks, threads, 0, stream>>>(wq_ptr, d_w_scale, op, k_i, n_i, row_stride, co_i);
      }
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) throw std::runtime_error(std::string("DequantWeightBf16: ") + cudaGetErrorString(err));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct DequantWeightBf16Op : Ort::CustomOpBase<DequantWeightBf16Op, DequantWeightBf16KernelState, true> {
  const char* GetName() const { return "DequantWeightBf16"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_OPTIONAL;
  }
  size_t GetOutputTypeCount() const { return skinny::kMaxOut; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_OPTIONAL;
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new DequantWeightBf16KernelState(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

template <bool FP8>
__global__ void __launch_bounds__(256) UntileHcBf16Kernel(const uint4* src, uint4* dst, const float* w_scale, int kind) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 204800) return;
  int dst_idx = 0;
  int col0 = 0;
  if (kind == 0) {
    const int rp = idx & 127;
    const int t1 = idx >> 7;
    const int i = t1 % 10;
    const int t2 = t1 / 10;
    const int j = t2 % 20;
    const int s = t2 / 20;
    dst_idx = (((s * 10 + i) * 128 + rp) * 20 + j);
    col0 = j * 16;
  } else {
    const int c = idx & 3;
    const int t1 = idx >> 2;
    const int rp = t1 & 31;
    const int t2 = t1 >> 5;
    const int i = t2 % 10;
    const int hb = t2 / 10;
    dst_idx = (((i * 32 + rp) * 4 + c) * 160 + hb);
    col0 = (c * 160 + hb) * 16;
  }
  if constexpr (!FP8) {
    dst[2 * dst_idx + 0] = src[2 * idx + 0];
    dst[2 * dst_idx + 1] = src[2 * idx + 1];
  } else {
    const uint4 raw16 = __ldg(src + idx);
    dst[2 * dst_idx + 0] = DeqFp8x8ToBf16U4(make_uint2(raw16.x, raw16.y), w_scale, col0);
    dst[2 * dst_idx + 1] = DeqFp8x8ToBf16U4(make_uint2(raw16.z, raw16.w), w_scale, col0 + 8);
  }
}

struct UntileHcBf16KernelState {
  const OrtApi& api;
  int kind = 0;
  float* d_w_scale = nullptr;
  size_t w_scale_len = 0;
  UntileHcBf16KernelState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    kind = static_cast<int>(k.GetAttribute<int64_t>("kind"));
    std::vector<float> scale_host;
    try {
      scale_host = k.GetAttributes<float>("w_scale");
    } catch (const std::exception&) {
    }
    if (!scale_host.empty()) {
      w_scale_len = scale_host.size();
      if (cudaMalloc(&d_w_scale, w_scale_len * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_w_scale, scale_host.data(), w_scale_len * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w_scale);
        d_w_scale = nullptr;
        throw std::runtime_error("UntileHcBf16: failed to upload w_scale");
      }
    }
  }
  ~UntileHcBf16KernelState() {
    cudaFree(d_w_scale);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto w = c.GetInput(0);
      auto winfo = w.GetTensorTypeAndShapeInfo();
      auto wshape = winfo.GetShape();
      const auto wtype = winfo.GetElementType();
      if (wshape != std::vector<int64_t>{160, 20480}) throw std::invalid_argument("UntileHcBf16: expected [160, 20480]");
      std::vector<int64_t> oshape = kind == 0 ? std::vector<int64_t>{10240, 320} : std::vector<int64_t>{320, 10240};
      auto out = c.GetOutput(0, oshape);
      cudaStream_t stream = Stream(api, raw);
      if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        UntileHcBf16Kernel<false><<<800, 256, 0, stream>>>(
            reinterpret_cast<const uint4*>(w.GetTensorRawData()),
            reinterpret_cast<uint4*>(out.GetTensorMutableRawData()), nullptr, kind);
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
        UntileHcBf16Kernel<true><<<800, 256, 0, stream>>>(
            reinterpret_cast<const uint4*>(w.GetTensorRawData()),
            reinterpret_cast<uint4*>(out.GetTensorMutableRawData()), d_w_scale, kind);
      } else {
        throw std::invalid_argument("UntileHcBf16: unsupported input element type");
      }
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) throw std::runtime_error(std::string("UntileHcBf16: ") + cudaGetErrorString(err));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct UntileHcBf16Op : Ort::CustomOpBase<UntileHcBf16Op, UntileHcBf16KernelState, true> {
  const char* GetName() const { return "UntileHcBf16"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 0 ? INPUT_OUTPUT_REQUIRED : INPUT_OUTPUT_OPTIONAL;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new UntileHcBf16KernelState(api, info);
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
    static DequantWeightBf16Op deq_op;
    static UntileHcBf16Op untile_op;
    static Ort::CustomOpDomain domain("qwen.fused_gemm");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&skinny_gemm);
      domain.Add(&shared_down);
      domain.Add(&argmax_rows);
      domain.Add(&deq_op);
      domain.Add(&untile_op);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
