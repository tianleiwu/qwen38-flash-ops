# ctest smoke tests. `ctest --output-on-failure` after building.
#  - load_<lib>:   dlopen + RegisterCustomOps symbol (no GPU, no ORT runtime needed)
#  - python_ops:   tests/test_load.py, only when a Python with onnxruntime-gpu, onnx and torch is found
#                  (-DQWEN38_PYTHON=/path/to/python); needs one CUDA GPU (CUDA_VISIBLE_DEVICES selects it).
add_executable(qwen38_load_check load_check.cpp)
target_link_libraries(qwen38_load_check PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(qwen38_load_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

foreach(lib IN LISTS QWEN38_OP_LIBRARIES)
  add_test(NAME load_${lib} COMMAND qwen38_load_check $<TARGET_FILE:${lib}>)
endforeach()

set(QWEN38_PYTHON "" CACHE FILEPATH "Python with onnxruntime-gpu, onnx and torch (enables the python_ops test)")
if(QWEN38_BUILD_ORT_FORKS)
  add_executable(qwen38_qsa_topk_test test_qsa_topk.cu)
  target_include_directories(qwen38_qsa_topk_test PRIVATE "${ORT_SOURCE_DIR}/onnxruntime")
  target_compile_options(qwen38_qsa_topk_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${QWEN38_CUDA_ARCH}>")
  add_test(NAME qsa_topk COMMAND qwen38_qsa_topk_test)
  set_tests_properties(qsa_topk PROPERTIES TIMEOUT 600)
endif()

if(QWEN38_PYTHON)
  add_test(NAME python_ops COMMAND "${QWEN38_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_load.py" --lib-dir "${CMAKE_BINARY_DIR}/lib")
  set_tests_properties(python_ops PROPERTIES TIMEOUT 600)
endif()
