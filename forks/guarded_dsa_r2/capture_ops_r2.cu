#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include "private_dsa.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace native = onnxruntime::contrib::cuda::static_capture;

namespace {

constexpr size_t kMaxRows = 4;

void CheckCuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// composite == nullptr (S=1 / prefill): one flag. Row chain (R rows, composite != nullptr): row r is the S=1 call
// at committed + r, so it needs committed + r + 1 <= capacity and seqlens_k == committed + R - 1; flags are per
// row and the status output is the composite: row r's code in nibble r.
__global__ void CheckCursorKernel(const int32_t* lengths, const int32_t* committed,
                                  int sequence, int capacity, int32_t* error, int32_t* composite) {
  if (composite == nullptr) {
    const int64_t total = static_cast<int64_t>(*committed) + sequence;
    if (*committed < 0 || total > capacity || *lengths != total - 1) atomicCAS(error, 0, 9);
    return;
  }
  int32_t status = 0;
  for (int r = 0; r < sequence; ++r) {
    const int64_t total = static_cast<int64_t>(*committed) + r + 1;
    if (*committed < 0 || total > capacity || *lengths != *committed + sequence - 1) atomicCAS(error + r, 0, 9);
    status |= (error[r] & 0xF) << (4 * r);
  }
  composite[0] = status;
}

struct SharedDsaPrefillWorkspace {
  std::mutex mu;
  __nv_bfloat16* prepared = nullptr;
  size_t prepared_elems = 0;
  uint32_t* bitmap = nullptr;
  size_t bitmap_elems = 0;
  float* workspace = nullptr;
  size_t workspace_elems = 0;

  void Ensure(size_t prep_n, size_t bit_n, size_t ws_n, __nv_bfloat16** out_prep, uint32_t** out_bit, float** out_ws) {
    if (prep_n > prepared_elems) {
      if (prepared) cudaFree(prepared);
      prepared = nullptr;
      prepared_elems = 0;
      CheckCuda(cudaMalloc(&prepared, prep_n * sizeof(__nv_bfloat16)));
      prepared_elems = prep_n;
    }
    if (bit_n > bitmap_elems) {
      if (bitmap) cudaFree(bitmap);
      bitmap = nullptr;
      bitmap_elems = 0;
      CheckCuda(cudaMalloc(&bitmap, bit_n * sizeof(uint32_t)));
      bitmap_elems = bit_n;
    }
    if (ws_n > workspace_elems) {
      if (workspace) cudaFree(workspace);
      workspace = nullptr;
      workspace_elems = 0;
      CheckCuda(cudaMalloc(&workspace, ws_n * sizeof(float)));
      workspace_elems = ws_n;
    }
    *out_prep = prepared;
    *out_bit = bitmap;
    *out_ws = workspace;
  }
};

SharedDsaPrefillWorkspace& GetSharedDsaPrefillWorkspace() {
  static SharedDsaPrefillWorkspace ws;
  return ws;
}

struct GuardedKernel {
  const OrtApi& api;
  onnxruntime::contrib::DynamicSparseAttentionParameters parameters;
  cudaDeviceProp properties{};
  __nv_bfloat16* prepared = nullptr;
  uint32_t* bitmap = nullptr;
  float* workspace = nullptr;
  int32_t* row_errors = nullptr;
  float* d_k_scale = nullptr;
  float* d_v_scale = nullptr;

  GuardedKernel(const OrtApi& supplied, const OrtKernelInfo* info) : api(supplied) {
    Ort::ConstKernelInfo metadata(info);
    const auto heads = metadata.GetAttribute<int64_t>("num_heads");
    const auto kv_heads = metadata.GetAttribute<int64_t>("kv_num_heads");
    const auto width = metadata.GetAttribute<int64_t>("head_size");
    const auto capacity = metadata.GetAttribute<int64_t>("cache_capacity");
    const auto budget = metadata.GetAttribute<int64_t>("selected_capacity");
    if (heads <= 0 || heads > 128 || kv_heads <= 0 || heads % kv_heads != 0 ||
        width <= 0 || width > 256 || capacity < 128 || capacity > 1048576 || budget < 0 || budget > 4096) {
      throw std::invalid_argument("GuardedDSA: unsupported explicit geometry");
    }
    parameters.batch_size = 1;
    parameters.sequence_length = 128;
    parameters.num_heads = static_cast<int>(heads);
    parameters.kv_num_heads = static_cast<int>(kv_heads);
    parameters.head_size = static_cast<int>(width);
    parameters.query_hidden_size = static_cast<int>(heads * width);
    parameters.kv_hidden_size = static_cast<int>(kv_heads * width);
    parameters.cache_capacity = static_cast<int>(capacity);
    parameters.past_cache_capacity = static_cast<int>(capacity);
    parameters.total_sequence_length = static_cast<int>(capacity);
    parameters.max_selected = static_cast<int>(budget);
    parameters.scale = metadata.GetAttribute<float>("scale");
    if (!std::isfinite(parameters.scale) || parameters.scale < 0) {
      throw std::invalid_argument("GuardedDSA: invalid scale");
    }
    std::vector<float> k_scale_host, v_scale_host;
    try {
      k_scale_host = metadata.GetAttributes<float>("k_scale");
      v_scale_host = metadata.GetAttributes<float>("v_scale");
    } catch (const std::exception&) {
    }
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    CheckCuda(cudaGetDeviceProperties(&properties, device));
    try {
      if (!k_scale_host.empty() || !v_scale_host.empty()) {
        if (k_scale_host.size() != static_cast<size_t>(parameters.kv_hidden_size) ||
            v_scale_host.size() != static_cast<size_t>(parameters.kv_hidden_size)) {
          throw std::invalid_argument("GuardedDSA: k_scale/v_scale length must match kv_num_heads * head_size");
        }
        const size_t scale_bytes = k_scale_host.size() * sizeof(float);
        CheckCuda(cudaMalloc(&d_k_scale, scale_bytes));
        CheckCuda(cudaMalloc(&d_v_scale, scale_bytes));
        CheckCuda(cudaMemcpy(d_k_scale, k_scale_host.data(), scale_bytes, cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(d_v_scale, v_scale_host.data(), scale_bytes, cudaMemcpyHostToDevice));
      }
      CheckCuda(cudaMalloc(&prepared, 128 * parameters.query_hidden_size * sizeof(__nv_bfloat16)));
      size_t bitmap_elements = 0;
      size_t workspace_elements = 0;
      auto workspace_parameters = parameters;
      for (int sequence = 1; sequence <= 128; ++sequence) {
        workspace_parameters.sequence_length = sequence;
        bitmap_elements = std::max(bitmap_elements, native::GetDynamicSparseAttentionValidationWorkspaceSize(
            workspace_parameters, properties.sharedMemPerBlock));
        workspace_elements = std::max(workspace_elements, native::GetDynamicSparseAttentionWorkspaceSize(
            workspace_parameters, sizeof(__nv_bfloat16), properties.sharedMemPerBlock));
      }
      // the 2-row chain runs two single-row decode plans side by side
      workspace_parameters.sequence_length = 1;
      workspace_elements = std::max(workspace_elements, kMaxRows * native::GetDynamicSparseAttentionWorkspaceSize(
          workspace_parameters, sizeof(__nv_bfloat16), properties.sharedMemPerBlock));
      CheckCuda(cudaMalloc(&row_errors, kMaxRows * sizeof(int32_t)));
      if (bitmap_elements) CheckCuda(cudaMalloc(&bitmap, bitmap_elements * sizeof(uint32_t)));
      if (workspace_elements) CheckCuda(cudaMalloc(&workspace, workspace_elements * sizeof(float)));
    } catch (...) {
      cudaFree(d_k_scale);
      cudaFree(d_v_scale);
      cudaFree(prepared);
      cudaFree(bitmap);
      cudaFree(workspace);
      cudaFree(row_errors);
      throw;
    }
  }

  ~GuardedKernel() {
    cudaFree(d_k_scale);
    cudaFree(d_v_scale);
    cudaFree(row_errors);
    cudaFree(prepared);
    cudaFree(bitmap);
    cudaFree(workspace);
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context(raw_context);
      const auto shape = context.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 4096 ||
          shape[2] != parameters.query_hidden_size) throw std::invalid_argument("GuardedDSA: query shape");
      auto params = parameters;
      params.sequence_length = static_cast<int>(shape[1]);
      const std::vector<int64_t> kv_shape{1, params.kv_num_heads, params.cache_capacity, params.head_size};
      const std::vector<int64_t> current_shape{1, shape[1], params.kv_hidden_size};
      for (int index : {1, 2}) {
        if (context.GetInput(index).GetTensorTypeAndShapeInfo().GetShape() != current_shape) {
          throw std::invalid_argument("GuardedDSA: current key/value shape");
        }
      }
      for (int index : {3, 4}) {
        if (context.GetInput(index).GetTensorTypeAndShapeInfo().GetShape() != kv_shape) {
          throw std::invalid_argument("GuardedDSA: fixed KV shape");
        }
      }
      if (context.GetInput(5).GetTensorTypeAndShapeInfo().GetShape() !=
              std::vector<int64_t>{shape[1], params.max_selected} ||
          context.GetInput(6).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape[1]}) {
        throw std::invalid_argument("GuardedDSA: selected indices/counts shape");
      }
      for (int index : {7, 8}) {
        if (context.GetInput(index).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1}) {
          throw std::invalid_argument("GuardedDSA: length/cursor shape");
        }
      }
      auto output = context.GetOutput(0, shape);
      auto present_key = context.GetOutput(1, kv_shape);
      auto present_value = context.GetOutput(2, kv_shape);
      auto error_output = context.GetOutput(3, std::vector<int64_t>{1});
      native::DynamicSparseAttentionData<__nv_bfloat16> data;
      const auto bf16 = [&](int index) {
        return reinterpret_cast<const __nv_bfloat16*>(context.GetInput(index).GetTensorData<Ort::BFloat16_t>());
      };
      data.query = bf16(0);
      data.key = bf16(1);
      data.value = bf16(2);
      const auto kv_elem_type = context.GetInput(3).GetTensorTypeAndShapeInfo().GetElementType();
      if (kv_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        data.kv_quant_mode = 0;
        data.past_key = bf16(3);
        data.past_value = bf16(4);
        data.present_key = reinterpret_cast<__nv_bfloat16*>(present_key.GetTensorMutableData<Ort::BFloat16_t>());
        data.present_value = reinterpret_cast<__nv_bfloat16*>(present_value.GetTensorMutableData<Ort::BFloat16_t>());
        if (data.past_key != data.present_key || data.past_value != data.present_value) {
          throw std::invalid_argument("GuardedDSA: caller must bind owned in-place KV outputs");
        }
      } else if (kv_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                 kv_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
        if (!d_k_scale || !d_v_scale) {
          throw std::invalid_argument("GuardedDSA: quantized KV requires k_scale and v_scale attributes");
        }
        data.kv_quant_mode = (kv_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) ? 1 : 2;
        data.past_key_q = static_cast<const uint8_t*>(context.GetInput(3).GetTensorRawData());
        data.past_value_q = static_cast<const uint8_t*>(context.GetInput(4).GetTensorRawData());
        data.present_key_q = static_cast<uint8_t*>(present_key.GetTensorMutableRawData());
        data.present_value_q = static_cast<uint8_t*>(present_value.GetTensorMutableRawData());
        data.k_scale = d_k_scale;
        data.v_scale = d_v_scale;
        if (data.past_key_q != data.present_key_q || data.past_value_q != data.present_value_q) {
          throw std::invalid_argument("GuardedDSA: caller must bind owned in-place quantized KV outputs");
        }
      } else {
        throw std::invalid_argument("GuardedDSA: unsupported KV cache element type");
      }
      data.selected_indices = context.GetInput(5).GetTensorData<int32_t>();
      data.selected_counts = context.GetInput(6).GetTensorData<int32_t>();
      data.seqlens_k = context.GetInput(7).GetTensorData<int32_t>();
      __nv_bfloat16* active_prep = prepared;
      uint32_t* active_bitmap = bitmap;
      float* active_ws = workspace;
      if (params.sequence_length > 128) {
        auto& sws = GetSharedDsaPrefillWorkspace();
        std::lock_guard<std::mutex> lock(sws.mu);
        const size_t prep_n = static_cast<size_t>(params.sequence_length) * params.query_hidden_size;
        const size_t bit_n = native::GetDynamicSparseAttentionValidationWorkspaceSize(
            params, properties.sharedMemPerBlock);
        const size_t ws_n = native::GetDynamicSparseAttentionWorkspaceSize(
            params, sizeof(__nv_bfloat16), properties.sharedMemPerBlock);
        sws.Ensure(prep_n, bit_n, ws_n, &active_prep, &active_bitmap, &active_ws);
      }
      data.prepared_query = active_prep;
      data.output = reinterpret_cast<__nv_bfloat16*>(output.GetTensorMutableData<Ort::BFloat16_t>());
      data.attention_workspace = active_ws;
      auto error = error_output.GetTensorMutableData<int32_t>();
      // 2 rows: chained through the S=1 decode path with per-row flags (scratch) and a composite status output
      const bool row_chain = params.sequence_length > 1 && static_cast<size_t>(params.sequence_length) <= kMaxRows;
      data.row_chain = row_chain ? params.sequence_length : 1;
      int32_t* flags = row_chain ? row_errors : error;
      data.validation_error = flags;
      data.composite_error = row_chain ? error : nullptr;
      const int32_t* committed_ptr = context.GetInput(8).GetTensorData<int32_t>();
      void* raw_stream = nullptr;
      Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(raw_context, &raw_stream));
      auto stream = static_cast<cudaStream_t>(raw_stream);
      auto status = native::ValidateDynamicSparseAttentionOnDevice(stream, data.selected_indices,
          data.selected_counts, data.seqlens_k, nullptr, params, flags, active_bitmap,
          properties.sharedMemPerBlock, false, row_chain, committed_ptr);
      if (!status.IsOK()) return api.CreateStatus(ORT_FAIL, status.ErrorMessage().c_str());
      status = native::LaunchDynamicSparseAttention<__nv_bfloat16>(stream, params, data, false, false,
          properties.maxThreadsPerBlock, properties.sharedMemPerBlock);
      if (!status.IsOK()) return api.CreateStatus(ORT_FAIL, status.ErrorMessage().c_str());
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
    }
  }
};

template <ONNXTensorElementDataType KvType>
struct GuardedOpTyped : Ort::CustomOpBase<GuardedOpTyped<KvType>, GuardedKernel, true> {
  const char* GetName() const { return "GuardedDSA"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 9; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 3 || index == 4) return KvType;
    return index < 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  size_t GetOutputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1 || index == 2) return KvType;
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new GuardedKernel(api, info); return nullptr; }
    catch (const std::exception& error) { return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what()); }
  }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static GuardedOpTyped<ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16> op_bf16;
    static GuardedOpTyped<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8> op_int8;
    static GuardedOpTyped<ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN> op_fp8;
    static Ort::CustomOpDomain domain("qwen.static");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&op_bf16);
      domain.Add(&op_int8);
      domain.Add(&op_fp8);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_FAIL, error.what());
  }
}