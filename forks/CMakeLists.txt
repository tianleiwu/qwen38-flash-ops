# Libraries that fork ONNX Runtime kernels and therefore need ORT *private* headers and the ORT static libraries.
# Only built with -DQWEN38_BUILD_ORT_FORKS=ON. See forks/README.md.
#
#   -DORT_SOURCE_DIR=<ORT source checkout>          (private headers: onnxruntime/, include/)
#   -DORT_BUILD_DIR=<ORT Release build tree>        (onnxruntime_config.h, libonnxruntime_common.a and its _deps/*-build static libs)
#   -DORT_DEPS_DIR=<dir with the third-party sources>   either the upper-case FETCHCONTENT_SOURCE_DIR layout (ABSEIL_CPP/, CUTLASS/, ...)
#                                                       or empty to use <ORT_BUILD_DIR>/_deps/<name>-src
#   -DCUDAToolkit_ROOT / CMAKE_CUDA_COMPILER = the CUDA toolkit the ORT build tree was compiled with (13.0)
foreach(v ORT_SOURCE_DIR ORT_BUILD_DIR)
  if(NOT ${v})
    message(FATAL_ERROR "-D${v}=... is required with QWEN38_BUILD_ORT_FORKS=ON (see forks/README.md)")
  endif()
endforeach()
if(NOT EXISTS "${ORT_BUILD_DIR}/libonnxruntime_common.a")
  message(FATAL_ERROR "${ORT_BUILD_DIR}/libonnxruntime_common.a not found: ORT_BUILD_DIR must be a built ORT tree")
endif()

function(qwen38_find_dep OUT UPPER LOWER)
  foreach(c "${ORT_DEPS_DIR}/${UPPER}" "${ORT_BUILD_DIR}/_deps/${LOWER}-src")
    if(EXISTS "${c}")
      set(${OUT} "${c}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "third-party source ${UPPER} (${LOWER}-src) not found; set ORT_DEPS_DIR")
endfunction()

qwen38_find_dep(DEP_CPUINFO   PYTORCH_CPUINFO  pytorch_cpuinfo)
qwen38_find_dep(DEP_DEEPGEMM  DEEP_GEMM        deep_gemm)
qwen38_find_dep(DEP_EIGEN     EIGEN3           eigen)
qwen38_find_dep(DEP_CUTLASS   CUTLASS          cutlass)
qwen38_find_dep(DEP_ABSEIL    ABSEIL_CPP       abseil_cpp)
qwen38_find_dep(DEP_DATE      DATE             date)
qwen38_find_dep(DEP_ONNX      ONNX             onnx)
qwen38_find_dep(DEP_PROTOBUF  PROTOBUF         protobuf)
qwen38_find_dep(DEP_FLATBUF   FLATBUFFERS      flatbuffers)
qwen38_find_dep(DEP_SAFEINT   SAFEINT          safeint)
qwen38_find_dep(DEP_GSL       GSL              gsl)
qwen38_find_dep(DEP_CUDNNFE   CUDNN_FRONTEND   cudnn_frontend)
qwen38_find_dep(DEP_MP11      MP11             mp11)

get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_ROOT "${CUDA_BIN_DIR}" DIRECTORY)

set(ORT_PRIVATE_INCLUDES
  "${ORT_SOURCE_DIR}/include/onnxruntime"
  "${ORT_SOURCE_DIR}/include/onnxruntime/core/session"
  "${DEP_CPUINFO}/include"
  "${ORT_BUILD_DIR}"
  "${ORT_SOURCE_DIR}/onnxruntime"
  "${DEP_DEEPGEMM}/deep_gemm/include"
  "${ORT_SOURCE_DIR}/include"
  "${DEP_EIGEN}"
  "${DEP_CUTLASS}/include"
  "${DEP_CUTLASS}/examples"
  "${DEP_CUTLASS}/tools/util/include"
  "${DEP_ABSEIL}"
  "${DEP_DATE}/include"
  "${DEP_ONNX}"
  "${ORT_BUILD_DIR}/_deps/onnx-build"
  "${DEP_PROTOBUF}/src"
  "${DEP_PROTOBUF}/third_party/utf8_range"
  "${DEP_FLATBUF}/include")
set(ORT_PRIVATE_SYSTEM_INCLUDES
  "${CUDA_ROOT}/targets/x86_64-linux/include"
  "${CUDA_ROOT}/targets/x86_64-linux/include/cccl"
  "$ENV{HOME}/cudnn_9.23_cuda13/include"
  "${DEP_SAFEINT}" "${DEP_GSL}/include" "${DEP_CUDNNFE}/include" "${DEP_MP11}/include")

# Compile definitions and nvcc flags of ORT's onnxruntime_providers_cuda_plugin target (the build these forks were developed against).
set(ORT_PRIVATE_DEFS
  BUILD_CUDA_EP_AS_PLUGIN COMPILE_HOPPER_TMA_GEMMS COMPILE_HOPPER_TMA_GROUPED_GEMMS CPUINFO_SUPPORTED CPUINFO_SUPPORTED_PLATFORM=1
  EIGEN_MPL2_ONLY EIGEN_USE_THREADS ENABLE_BF16 ENABLE_CUDA_NHWC_OPS ENABLE_DLPACK ENABLE_FP4 ENABLE_FP8
  EXCLUDE_SM_100 EXCLUDE_SM_110 EXCLUDE_SM_120 EXCLUDE_SM_80 EXCLUDE_SM_86 EXCLUDE_SM_89 HAS_SM80_OR_LATER HAS_SM90_OR_LATER
  NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING ONLY_C_LOCALE=0 ONNX_ML=1 ONNX_NAMESPACE=onnx ONNX_USE_LITE_PROTO=1 ORT_API_MANUAL_INIT
  ORT_ENABLE_GQA_VALUE_LAYOUT ORT_ENABLE_STREAM "ORT_PLUGIN_EP_MIN_ORT_VERSION=\"1.24.4\"" "ORT_PLUGIN_EP_VERSION=\"1.31.0-dev\""
  ORT_USE_EP_API_ADAPTERS=1 PLATFORM_POSIX USE_COMPACT_FPA_INTB_GEMM=1 USE_CUDA=1 USE_DEEP_GEMM USE_DX_INTEROP=0
  USE_FLASH_ATTENTION=1 USE_FP4_QMOE USE_FP8_KV_CACHE=1 USE_FPA_INTB_GEMM=1 USE_INT4_KV_CACHE=1 USE_MEMORY_EFFICIENT_ATTENTION=1
  USE_TRT_FUSED_ATTENTION=1 _GNU_SOURCE __ONNX_NO_DOC_STRINGS NDEBUG)
set(ORT_PRIVATE_NVCC_FLAGS
  -forward-unknown-to-host-compiler -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong
  -pipe -fstack-clash-protection -fcf-protection -Xfatbin=-compress-all -compress-mode=size --expt-relaxed-constexpr
  "SHELL:--Werror default-stream-launch" "SHELL:-Xcudafe --diag_suppress=bad_friend_decl"
  "SHELL:-Xcudafe --diag_suppress=unsigned_compare_with_zero" "SHELL:-Xcudafe --diag_suppress=expr_has_no_effect"
  "SHELL:--generate-code=arch=compute_90a,code=[sm_90a]" -Xcompiler=-fPIC -Wno-deprecated-declarations
  "SHELL:-Xcudafe --diag_suppress=conversion_function_not_usable" "SHELL:--compiler-options -Wall"
  "SHELL:--compiler-options -Wno-deprecated-copy" "SHELL:--compiler-options -Wno-nonnull-compare"
  "SHELL:--compiler-options -Wno-interference-size" "SHELL:-Xcudafe --diag_suppress=550" "SHELL:-Xcudafe --diag_suppress=2810"
  --diag-suppress=177 --static-global-template-stub=false --diag-suppress=221 --diag-suppress=1427 --diag-suppress=2908
  --diag-suppress=68 --diag-suppress=970 --diag-suppress=2189 -Xptxas=-w -DCUTLASS_ENABLE_GDC_FOR_SM90=1 "SHELL:--threads 4"
  "SHELL:-Werror all-warnings")

# ORT's own static libraries that the forks link (the `-Wl,--start-group` of the original build scripts).
set(ORT_STATIC_LIBS
  "${ORT_BUILD_DIR}/libonnxruntime_common.a"
  "${ORT_BUILD_DIR}/_deps/pytorch_cpuinfo-build/libcpuinfo.a"
  "${ORT_BUILD_DIR}/_deps/cpp_client_telemetry-build/lib/libmat.a"
  "${ORT_BUILD_DIR}/_deps/cpp_client_telemetry-build/lib/libsqlite3_bundled.a"
  "${ORT_BUILD_DIR}/_deps/cpp_client_telemetry-build/lib/libzlib_bundled.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_curl-build/lib/libcurl.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_mbedtls-build/library/libmbedtls.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_mbedtls-build/library/libmbedx509.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_mbedtls-build/library/libmbedcrypto.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_mbedtls-build/3rdparty/everest/libeverest.a"
  "${ORT_BUILD_DIR}/_deps/matsdk_mbedtls-build/3rdparty/p256-m/libp256m.a")

# cuda_runtime_status.cc: the one CUDA-runtime error-reporting symbol of ORT that the forks need (see the file).
add_library(qwen38_cuda_runtime_status OBJECT guarded_dsa_r2/cuda_runtime_status.cc)
target_compile_features(qwen38_cuda_runtime_status PRIVATE cxx_std_20)
target_include_directories(qwen38_cuda_runtime_status PRIVATE "${ORT_SOURCE_DIR}/onnxruntime" "${ORT_SOURCE_DIR}/include/onnxruntime" "${CUDA_ROOT}/include")
target_compile_options(qwen38_cuda_runtime_status PRIVATE -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS
  -fstack-protector-strong -pipe -fstack-clash-protection -fcf-protection -fvisibility=hidden -ffunction-sections -fdata-sections)
target_compile_definitions(qwen38_cuda_runtime_status PRIVATE NDEBUG)

function(qwen38_add_fork_library NAME EXPORTS)
  set(srcs ${ARGN})
  add_library(${NAME} SHARED ${srcs} $<TARGET_OBJECTS:qwen38_cuda_runtime_status>)
  set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX CUDA_ARCHITECTURES OFF CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(${NAME} PRIVATE ${ORT_PRIVATE_INCLUDES})
  target_include_directories(${NAME} SYSTEM PRIVATE ${ORT_PRIVATE_SYSTEM_INCLUDES})
  target_compile_definitions(${NAME} PRIVATE ${ORT_PRIVATE_DEFS})
  target_compile_options(${NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${ORT_PRIVATE_NVCC_FLAGS}>")
  target_link_options(${NAME} PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,defs
    -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/${EXPORTS})
  target_link_directories(${NAME} PRIVATE "${CUDA_ROOT}/lib64")
  target_link_libraries(${NAME} PRIVATE -Wl,--start-group ${ORT_STATIC_LIBS} -Wl,--end-group cudart dl pthread)
  set(QWEN38_OP_LIBRARIES ${QWEN38_OP_LIBRARIES} ${NAME} PARENT_SCOPE)
endfunction()

# libqwen_guarded_dsa_r2_v12.so  (domain qwen.static, op GuardedDSA)
qwen38_add_fork_library(qwen_guarded_dsa_r2_v12 guarded_dsa_r2/guarded.exports
  guarded_dsa_r2/capture_ops_r2.cu guarded_dsa_r2/guarded_dsa_r2.cu)
target_include_directories(qwen_guarded_dsa_r2_v12 BEFORE PRIVATE guarded_dsa_r2)

# libqwen_packed_qsa_indexer_v11.so  (domain qwen.capture, op PackedQSAIndexer)
qwen38_add_fork_library(qwen_packed_qsa_indexer_v11 packed_qsa_v11/packed_qsa.exports
  packed_qsa_v11/packed_qsa_op_v11.cu packed_qsa_v11/packed_sparse_attention_indexer_impl.cu)
target_include_directories(qwen_packed_qsa_indexer_v11 BEFORE PRIVATE packed_qsa_v11)
target_compile_definitions(qwen_packed_qsa_indexer_v11 PRIVATE PACKED_QSA_EXACT_FLOAT_STATE)

set(QWEN38_OP_LIBRARIES ${QWEN38_OP_LIBRARIES} PARENT_SCOPE)
