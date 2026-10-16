#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include <packed_sparse_attention_indexer_impl.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace packed = onnxruntime::contrib::cuda;

namespace {

using QsaState = packed::QsaStateType<__nv_bfloat16>;

void CheckCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void RequireShape(
    const Ort::ConstValue& value,
    const std::vector<int64_t>& expected,
    const char* name) {
  if (value.GetTensorTypeAndShapeInfo().GetShape() != expected) {
    throw std::invalid_argument(std::string("PackedQSAIndexer: invalid ") + name);
  }
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  stream << "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      stream << ",";
    }
    stream << shape[index];
  }
  stream << "]";
  return stream.str();
}

struct PackedQsaKernel {
  const OrtApi& api;
  int compress_ratio;
  int state_capacity;
  int token_budget;
  int num_heads;
  int head_size;
  int max_tokens;
  float epsilon;
  float scale;
  float* workspace = nullptr;
  int instance_id;

  PackedQsaKernel(const OrtApi& supplied, const OrtKernelInfo* info) : api(supplied) {
    static std::atomic<int> next_id{0};
    instance_id = next_id.fetch_add(1);
    Ort::ConstKernelInfo metadata(info);
    compress_ratio = static_cast<int>(metadata.GetAttribute<int64_t>("compress_ratio"));
    state_capacity = static_cast<int>(metadata.GetAttribute<int64_t>("state_capacity"));
    token_budget = static_cast<int>(metadata.GetAttribute<int64_t>("token_budget"));
    num_heads = static_cast<int>(metadata.GetAttribute<int64_t>("num_heads"));
    head_size = static_cast<int>(metadata.GetAttribute<int64_t>("head_size"));
    max_tokens = static_cast<int>(metadata.GetAttribute<int64_t>("max_tokens"));
    epsilon = metadata.GetAttribute<float>("epsilon");
    scale = metadata.GetAttribute<float>("scale");
    if (compress_ratio <= 0 || state_capacity <= 0 || token_budget <= 0 ||
        token_budget % compress_ratio != 0 || num_heads <= 0 ||
        head_size <= 0 || max_tokens < 1 || max_tokens > 128 ||
        !std::isfinite(epsilon) || epsilon < 0 ||
        !std::isfinite(scale) || scale <= 0) {
      throw std::invalid_argument("PackedQSAIndexer: invalid attributes");
    }
    packed::PackedSparseAttentionIndexerParams params;
    params.batch_size = 1;
    params.total_tokens = max_tokens;
    params.num_heads = num_heads;
    params.head_size = head_size;
    params.query_row_stride = (num_heads + 1) * head_size;
    params.key_row_stride = params.query_row_stride;
    params.rotary_width = head_size / 4;
    params.max_rotary_length = 262144;
    params.compress_ratio = compress_ratio;
    params.state_capacity = state_capacity;
    params.buffer_capacity = 2 * compress_ratio - 1;
    params.capacity = token_budget + compress_ratio - 1;
    params.has_position_ids = true;
    params.epsilon = epsilon;
    params.scale = scale;
    params.block_topk = token_budget / compress_ratio;
    const size_t workspace_count =
        packed::GetQsaPackedWorkspaceFloatCount(params);
    CheckCuda(cudaMalloc(&workspace, workspace_count * sizeof(float)));
  }

  ~PackedQsaKernel() {
    cudaFree(workspace);
  }

  void* OptionalOutput(OrtKernelContext* raw_context, size_t index, const std::vector<int64_t>& shape) {
    OrtValue* value = nullptr;
    Ort::ThrowOnError(api.KernelContext_GetOutput(raw_context, index, shape.data(), shape.size(), &value));
    if (value == nullptr) return nullptr;
    void* data = nullptr;
    Ort::ThrowOnError(api.GetTensorMutableData(value, &data));
    return data;
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context(raw_context);
      const auto query_shape =
          context.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
      if (query_shape.size() != 2 || query_shape[0] < 1 ||
          query_shape[0] > max_tokens ||
          query_shape[1] != (num_heads + 1) * head_size) {
        throw std::invalid_argument("PackedQSAIndexer: invalid query");
      }
      const int total_tokens = static_cast<int>(query_shape[0]);
      RequireShape(context.GetInput(2), {head_size}, "query norm");
      RequireShape(context.GetInput(3), {head_size}, "key norm");
      const auto cos_shape =
          context.GetInput(4).GetTensorTypeAndShapeInfo().GetShape();
      const auto sin_shape =
          context.GetInput(5).GetTensorTypeAndShapeInfo().GetShape();
      const bool unbatched_cache =
          cos_shape.size() == 2 && cos_shape[0] > 0 &&
          cos_shape[1] > 0 && cos_shape[1] <= head_size &&
          cos_shape[1] % 2 == 0;
      const bool batched_cache =
          cos_shape.size() == 3 && cos_shape[0] == 1 &&
          cos_shape[1] > 0 && cos_shape[2] > 0 &&
          cos_shape[2] <= head_size && cos_shape[2] % 2 == 0;
      if ((!unbatched_cache && !batched_cache) || sin_shape != cos_shape) {
        throw std::invalid_argument(
            "PackedQSAIndexer: invalid rotary cache cos=" +
            ShapeString(cos_shape) + " sin=" + ShapeString(sin_shape));
      }
      RequireShape(context.GetInput(6), {2}, "cumulative lengths");
      RequireShape(context.GetInput(7), {1}, "past lengths");
      const auto position_shape =
          context.GetInput(11).GetTensorTypeAndShapeInfo().GetShape();
      if (
          position_shape != std::vector<int64_t>{total_tokens} &&
          position_shape != std::vector<int64_t>{1, total_tokens}) {
        throw std::invalid_argument("PackedQSAIndexer: invalid position ids");
      }
      RequireShape(
          context.GetInput(12), {1, state_capacity, head_size}, "key state");
      const int buffer_capacity = 2 * compress_ratio - 1;
      RequireShape(
          context.GetInput(13), {1, buffer_capacity, head_size}, "pending buffer");
      RequireShape(context.GetInput(15), {1, 2}, "state lengths");

      const int selected_capacity = token_budget + compress_ratio - 1;
      auto selected_indices =
          context.GetOutput(0, {total_tokens, selected_capacity});
      auto selected_counts = context.GetOutput(1, {total_tokens});
      auto present_key =
          context.GetOutput(2, {1, state_capacity, head_size});
      auto present_buffer =
          context.GetOutput(3, {1, buffer_capacity, head_size});
      auto present_lengths = context.GetOutput(5, {1, 2});
      auto validation_status = context.GetOutput(6, {1});
      // optional outputs 7/8 (after row 0), 9/10 (after row 1), 11/12 (after row 2)
      void* mid_buffer[3] = {nullptr, nullptr, nullptr};
      void* mid_lengths[3] = {nullptr, nullptr, nullptr};
      for (int m = 0; m < 3; ++m) {
        const size_t idx_buf = 7 + 2 * m;
        const size_t idx_len = 8 + 2 * m;
        if (context.GetOutputCount() > idx_buf) {
          mid_buffer[m] = OptionalOutput(raw_context, idx_buf, {1, buffer_capacity, head_size});
        }
        if (context.GetOutputCount() > idx_len) {
          mid_lengths[m] = OptionalOutput(raw_context, idx_len, {1, 2});
        }
        if ((mid_buffer[m] == nullptr) != (mid_lengths[m] == nullptr)) {
          throw std::invalid_argument("PackedQSAIndexer: mid buffer and mid lengths must be connected together");
        }
      }

      packed::PackedSparseAttentionIndexerParams params;
      params.batch_size = 1;
      params.total_tokens = total_tokens;
      params.num_heads = num_heads;
      params.head_size = head_size;
      params.query_row_stride = (num_heads + 1) * head_size;
      params.key_row_stride = params.query_row_stride;
      params.rotary_width =
          static_cast<int>(cos_shape[cos_shape.size() - 1]);
      params.max_rotary_length =
          static_cast<int>(cos_shape[cos_shape.size() - 2]);
      params.cos_cache_batched = batched_cache;
      params.compress_ratio = compress_ratio;
      params.state_capacity = state_capacity;
      params.buffer_capacity = buffer_capacity;
      params.capacity = selected_capacity;
      params.has_position_ids = true;
      params.epsilon = epsilon;
      params.scale = scale;
      params.block_topk = token_budget / compress_ratio;
      params.mid_kv_buffer = mid_buffer[0];
      params.mid_state_lengths = static_cast<int32_t*>(mid_lengths[0]);
      params.mid_kv_buffer_extra[0] = mid_buffer[1];
      params.mid_state_lengths_extra[0] = static_cast<int32_t*>(mid_lengths[1]);
      params.mid_kv_buffer_extra[1] = mid_buffer[2];
      params.mid_state_lengths_extra[1] = static_cast<int32_t*>(mid_lengths[2]);
      int rejected_mask = 0;
      const int n_rows = total_tokens < 1 ? 1 : (total_tokens > 8 ? 8 : static_cast<int>(total_tokens));
      for (int r = 0; r < n_rows; ++r) rejected_mask |= (9 << (4 * r));
      params.rejected_status_code = rejected_mask;

      void* raw_stream = nullptr;
      Ort::ThrowOnError(
          api.KernelContext_GetGPUComputeStream(raw_context, &raw_stream));
      auto stream = static_cast<cudaStream_t>(raw_stream);
      const auto bf16 = [&](int index) {
        return reinterpret_cast<const __nv_bfloat16*>(
            context.GetInput(index).GetTensorData<Ort::BFloat16_t>());
      };
      const __nv_bfloat16* query = bf16(0);
      if (std::getenv("PACKED_QSA_DEBUG_SYNC") != nullptr) {
        std::fprintf(
            stderr,
            "[packed-qsa] instance=%d tokens=%d launch\n",
            instance_id,
            total_tokens);
        std::fflush(stderr);
      }
      auto status = packed::LaunchQsaPackedSparseAttentionIndexer<__nv_bfloat16>(
          stream,
          params,
          query,
          query + num_heads * head_size,
          bf16(2),
          bf16(3),
          bf16(4),
          bf16(5),
          context.GetInput(6).GetTensorData<int32_t>(),
          context.GetInput(7).GetTensorData<int32_t>(),
          context.GetInput(11).GetTensorData<int64_t>(),
          context.GetInput(12).GetTensorData<QsaState>(),
          bf16(13),
          context.GetInput(15).GetTensorData<int32_t>(),
          nullptr,
          false,
          selected_indices.GetTensorMutableData<int32_t>(),
          selected_counts.GetTensorMutableData<int32_t>(),
          present_key.GetTensorMutableData<QsaState>(),
          reinterpret_cast<__nv_bfloat16*>(
              present_buffer.GetTensorMutableData<Ort::BFloat16_t>()),
          present_lengths.GetTensorMutableData<int32_t>(),
          nullptr,
          workspace,
          validation_status.GetTensorMutableData<int32_t>());
      if (!status.IsOK()) {
        return api.CreateStatus(ORT_FAIL, status.ErrorMessage().c_str());
      }
      CheckCuda(cudaGetLastError());
      if (std::getenv("PACKED_QSA_DEBUG_SYNC") != nullptr) {
        CheckCuda(cudaStreamSynchronize(stream));
        std::fprintf(
            stderr,
            "[packed-qsa] instance=%d tokens=%d complete\n",
            instance_id,
            total_tokens);
        std::fflush(stderr);
      }
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
    }
  }
};

struct PackedQsaOp : Ort::CustomOpBase<PackedQsaOp, PackedQsaKernel, true> {
  const char* GetName() const { return "PackedQSAIndexer"; }
  const char* GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  }
  size_t GetInputTypeCount() const { return 16; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 6 || index == 7 || index == 15) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    if (index == 11) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
#if defined(PACKED_QSA_EXACT_FLOAT_STATE)
    if (index == 12) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
#endif
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
      size_t index) const {
    return index == 1 || (index >= 8 && index <= 10) || index == 14
               ? INPUT_OUTPUT_OPTIONAL
               : INPUT_OUTPUT_REQUIRED;
  }
  size_t GetOutputTypeCount() const { return 13; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
#if defined(PACKED_QSA_EXACT_FLOAT_STATE)
    if (index == 2) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
#endif
    return index == 0 || index == 1 || index == 5 || index == 6 || index == 8 || index == 10 || index == 12
               ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
               : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(
      size_t index) const {
    return index == 4 || index >= 7 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
  OrtStatus* CreateKernelV2(
      const OrtApi& api,
      const OrtKernelInfo* info,
      void** kernel) const noexcept {
    try {
      *kernel = new PackedQsaKernel(api, info);
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
    }
  }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) {
    return api_base->GetApi(1)->CreateStatus(
        ORT_FAIL, "Unsupported ORT API");
  }
  Ort::InitApi(api);
  try {
    static PackedQsaOp operation;
    static Ort::CustomOpDomain domain("qwen.capture");
    static std::once_flag registered;
    std::call_once(registered, [&]() { domain.Add(&operation); });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_FAIL, error.what());
  }
}
