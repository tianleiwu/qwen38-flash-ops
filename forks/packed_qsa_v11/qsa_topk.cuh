// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cub/block/block_merge_sort.cuh>
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime::contrib::cuda::qsa_topk {

constexpr int kChunkSize = 2048;
constexpr int kTopK = 512;
constexpr int kThreads = 512;

namespace topk = onnxruntime::cuda::topk;

template <int ItemsPerThread>
using ChunkSort = cub::BlockMergeSort<uint64_t, kThreads, ItemsPerThread>;

template <int ItemsPerThread>
__device__ __forceinline__ void SortChunk(const uint64_t* input, uint64_t* output, int input_count,
                                         typename ChunkSort<ItemsPerThread>::TempStorage& temp) {
  // Reuse the packed stable-key CUB merge sort used by QMoE SoftmaxTopKMergeKernel
  // and GenAI FindPartitionTopK_StableSort_Merge. QSA's scoring stage has already
  // packed the keys; sorting them directly preserves score bits and index ties.
  uint64_t keys[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x * ItemsPerThread + item;
    keys[item] = index < input_count ? input[index] : topk::kPaddingSortKey;
  }
  ChunkSort<ItemsPerThread>(temp).Sort(keys, topk::Greater<uint64_t>());
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int rank = threadIdx.x * ItemsPerThread + item;
    if (rank < kTopK) output[rank] = keys[item];
  }
}

__global__ void QsaCompactTileTopKKernel(const uint64_t* input_keys, uint64_t* output_keys,
                                        int input_stride, int output_stride) {
  __shared__ union {
    ChunkSort<1>::TempStorage small;
    ChunkSort<2>::TempStorage medium;
    ChunkSort<4>::TempStorage large;
  } temp;
  const int chunk_start = static_cast<int>(blockIdx.x) * kChunkSize;
  const uint64_t* input = input_keys + static_cast<int64_t>(blockIdx.y) * input_stride + chunk_start;
  uint64_t* output = output_keys + static_cast<int64_t>(blockIdx.y) * output_stride + blockIdx.x * kTopK;
  const int count = min(kChunkSize, input_stride - chunk_start);

  // Scoring writes sorted 8- or 32-key tiles, with a contiguous prefix of live
  // tiles and zero padding afterwards. These probes are tile-aligned for both
  // layouts. The choice depends only on device data, so graph replay naturally
  // follows the growing context without recapture or host synchronization.
  if (input[0] == topk::kPaddingSortKey) {
    output[threadIdx.x] = topk::kPaddingSortKey;
  } else if (count > 1024 && input[1024] != topk::kPaddingSortKey) {
    SortChunk<4>(input, output, count, temp.large);
  } else if (count > 512 && input[512] != topk::kPaddingSortKey) {
    SortChunk<2>(input, output, min(count, 1024), temp.medium);
  } else {
    SortChunk<1>(input, output, min(count, 512), temp.small);
  }
}

}  // namespace onnxruntime::contrib::cuda::qsa_topk
