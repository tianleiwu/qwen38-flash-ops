// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../forks/packed_qsa_v11/qsa_topk.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace qsa = onnxruntime::contrib::cuda::qsa_topk;

#define CUDA_CHECK(call)                                                                          \
  do {                                                                                           \
    const auto error = (call);                                                                    \
    if (error != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));             \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

uint64_t Pack(float score, int index) {
  uint32_t bits;
  std::memcpy(&bits, &score, sizeof(bits));
  if (score == 0.0f) bits = 0;
  const uint32_t ordered = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
  return (static_cast<uint64_t>(ordered) << 32) | (0xffffffffu - index);
}

int main() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::mt19937 random(20261017);
  int cases = 0;
  for (int tile : {8, 32}) {
    for (int stride : {2048, 2048 + tile, 4096, 8288, 12320}) {
      for (int rows : {1, 2, 5, 64}) {
        const int chunks = (stride + qsa::kChunkSize - 1) / qsa::kChunkSize;
        const int output_stride = chunks * qsa::kTopK;
        std::vector<uint64_t> input(rows * stride), output(rows * output_stride), expected(output.size());
        uint64_t *device_input, *device_output;
        CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(uint64_t)));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        qsa::QsaCompactTileTopKKernel<<<dim3(chunks, rows), qsa::kThreads, 0, stream>>>(
            device_input, device_output, stride, output_stride);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        // Replay the same graph while growing/shrinking across all adaptive sort
        // boundaries, including empty rows and partial final score tiles/chunks.
        for (int live : {stride, 0, 1, 7, 8, 31, 32, 255, 256, 511, 512, 513,
                         1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, stride - 1}) {
          for (int pattern = 0; pattern < 4; ++pattern) {
            std::fill(input.begin(), input.end(), 0);
            std::fill(expected.begin(), expected.end(), 0);
            for (int row = 0; row < rows; ++row) {
              const int visible = std::min(stride, std::max(0, live - row % 3));
              for (int i = 0; i < visible; ++i) {
                float score = pattern == 0 ? static_cast<float>(random() % 4096) :
                              pattern == 1 ? 0.0f : pattern == 2 ? static_cast<float>(i) :
                              static_cast<float>((i % 17) - 8);
                input[row * stride + i] = Pack(score, i);
              }
              for (int offset = 0; offset < stride; offset += tile) {
                std::sort(input.begin() + row * stride + offset,
                          input.begin() + row * stride + std::min(offset + tile, stride), std::greater<uint64_t>());
              }
              for (int chunk = 0; chunk < chunks; ++chunk) {
                const auto begin = input.begin() + row * stride + chunk * qsa::kChunkSize;
                const auto end = input.begin() + row * stride + std::min((chunk + 1) * qsa::kChunkSize, stride);
                std::vector<uint64_t> keys(begin, end);
                std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
                std::copy_n(keys.begin(), std::min<int>(keys.size(), qsa::kTopK),
                            expected.begin() + row * output_stride + chunk * qsa::kTopK);
              }
            }
            CUDA_CHECK(cudaMemcpyAsync(device_input, input.data(), input.size() * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(device_output, 0xcd, output.size() * sizeof(uint64_t), stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaMemcpyAsync(output.data(), device_output, output.size() * sizeof(uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (output != expected) {
              std::fprintf(stderr, "FAIL tile=%d stride=%d rows=%d live=%d pattern=%d\n",
                           tile, stride, rows, live, pattern);
              return 1;
            }
            ++cases;
          }
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_output));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("PASS: %d exact CPU-sort comparisons with CUDA graph replay\n", cases);
}
