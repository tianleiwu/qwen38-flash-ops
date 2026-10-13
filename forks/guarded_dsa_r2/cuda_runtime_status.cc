#include <cuda_runtime.h>
#include <string>
#include <type_traits>

#include "core/common/status.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {

template <>
Status CudaCall<cudaError, false, cudaError>(
    cudaError error, const char* expression, const char* library,
    cudaError success, const char* detail, const char* file, const int line) {
  if (error == success) return common::Status::OK();
  std::string message = std::string(library) + " failure " + std::to_string(error) +
                        ": " + cudaGetErrorString(error) + "; " + expression +
                        "; " + detail + "; " + file + ":" + std::to_string(line);
  return common::Status(common::ONNXRUNTIME, common::FAIL, message);
}

}