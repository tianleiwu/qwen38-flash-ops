cmake_minimum_required(VERSION 3.22)

# The architecture flags are spelled out per target (see qwen38_add_op_library), so CMake must not add its own.
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(qwen38_flash_ops LANGUAGES CXX CUDA)

set(QWEN38_CUDA_ARCH "90" CACHE STRING "SM version of the generated code (the shipped libraries use 90; fused_attn uses the arch-specific 90a)")
option(QWEN38_BUILD_ORT_FORKS "Also build the libraries that fork ORT kernels (forks/, needs an ORT source + build tree)" OFF)
option(QWEN38_BUILD_TESTS "Add the ctest smoke tests" ON)

# Headers of the ORT custom-op API (onnxruntime_c_api.h, onnxruntime_cxx_api.h, ...).
# Any of: -DORT_INCLUDE_DIR=..., $ENV{ORT_INCLUDE_DIR}, $ENV{ORT_HOME}/include.
if(NOT ORT_INCLUDE_DIR AND DEFINED ENV{ORT_INCLUDE_DIR})
  set(ORT_INCLUDE_DIR "$ENV{ORT_INCLUDE_DIR}")
endif()
if(NOT ORT_INCLUDE_DIR AND DEFINED ENV{ORT_HOME})
  set(ORT_INCLUDE_DIR "$ENV{ORT_HOME}/include")
endif()
set(ORT_INCLUDE_DIR "${ORT_INCLUDE_DIR}" CACHE PATH "Directory containing onnxruntime_c_api.h and onnxruntime_cxx_api.h")
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ORT_INCLUDE_DIR must point at a directory containing onnxruntime_cxx_api.h "
    "(the include/ dir of an onnxruntime release tarball, or include/onnxruntime/core/session of an ORT source tree). "
    "Got: '${ORT_INCLUDE_DIR}'. See README.md.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# The shipped libraries were built with plain `nvcc -O3` (no -DNDEBUG): keep the flags identical.
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# One shared library per domain; the output name is lib<NAME>.so, identical to the file name the model packages load.
function(qwen38_add_op_library NAME ARCH_SUFFIX)
  add_library(${NAME} SHARED ${ARGN})
  target_include_directories(${NAME} PRIVATE "${ORT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  # "" -> -arch=sm_XX (SASS + PTX); "a" -> -gencode arch=compute_XXa,code=sm_XXa (SASS only), exactly as in the original builds.
  if(ARCH_SUFFIX STREQUAL "")
    set(_arch_flags "-arch=sm_${QWEN38_CUDA_ARCH}")
  else()
    set(_arch_flags "-gencode arch=compute_${QWEN38_CUDA_ARCH}${ARCH_SUFFIX},code=sm_${QWEN38_CUDA_ARCH}${ARCH_SUFFIX}")
  endif()
  target_compile_options(${NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${_arch_flags}>")
  set_target_properties(${NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  list(APPEND QWEN38_OP_LIBRARIES ${NAME})
  set(QWEN38_OP_LIBRARIES ${QWEN38_OP_LIBRARIES} PARENT_SCOPE)
endfunction()

qwen38_add_op_library(qwen_fused_ops    ""  src/fused_ops/fused_ops.cu)             # domain qwen.fused
qwen38_add_op_library(qwen_fused_attn_r5 a  src/fused_attn/fused_attn_r5.cu)        # domain qwen.fused_attn
qwen38_add_op_library(qwen_fused_hc_r5  ""  src/fused_hc/fused_hc_r5.cu)            # domain qwen.fused_hc
qwen38_add_op_library(qwen_fused_gemm_r5 "" src/fused_gemm/fused_gemm.cu)           # domain qwen.fused_gemm
target_include_directories(qwen_fused_gemm_r5 PRIVATE src/fused_gemm)
qwen38_add_op_library(qwen_fold         ""  src/fold/qwen_fold_ops.cu)              # domain qwen.fold
qwen38_add_op_library(qwen_mtp_head     ""  src/mtp_head/qwen_mtp_ops.cu)           # domain qwen.mtp

if(QWEN38_BUILD_ORT_FORKS)
  add_subdirectory(forks)
endif()

if(QWEN38_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
