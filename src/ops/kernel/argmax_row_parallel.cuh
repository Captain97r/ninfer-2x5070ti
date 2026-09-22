#pragma once

#include "ops/common/argmax_row_parallel_workspace.h"
#include "ops/kernel/peer_mailbox_device.cuh"

#include <cuda_bf16.h>
#include <climits>
#include <math_constants.h>

namespace ninfer::ops::detail {

__device__ __forceinline__ ArgmaxCandidate argmax_shard_empty() {
    return {-CUDART_INF_F, INT_MAX, {0, 0}};
}

// Preserve the existing full-column scan's row-zero seed. A NaN at any other row is
// ineligible; a NaN specifically at global row zero is the absorbing winner.
__device__ __forceinline__ bool argmax_shard_better(ArgmaxCandidate a, ArgmaxCandidate b) {
    if (b.index == 0 && isnan(b.value)) { return false; }
    if (a.index == 0 && isnan(a.value)) { return true; }
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

__device__ __forceinline__ ArgmaxCandidate argmax_shard_warp(ArgmaxCandidate best) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const ArgmaxCandidate other{__shfl_down_sync(0xffffffffu, best.value, offset),
                                    __shfl_down_sync(0xffffffffu, best.index, offset), {0, 0}};
        if (argmax_shard_better(other, best)) { best = other; }
    }
    return best;
}

__global__ __launch_bounds__(kArgmaxShardBlock) void argmax_shard_partial_kernel(
    const __nv_bfloat16* logits, ArgmaxCandidate* partial, int physical_rows, int local_valid,
    int global_offset, int tiles) {
    const int column = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.x) * kArgmaxShardBlock + threadIdx.x;
    ArgmaxCandidate best = argmax_shard_empty();
    if (row < local_valid) {
        const ArgmaxCandidate value{
            __bfloat162float(logits[static_cast<std::int64_t>(column) * physical_rows + row]),
            global_offset + row, {0, 0}};
        if (argmax_shard_better(value, best)) { best = value; }
    }
    best = argmax_shard_warp(best);
    __shared__ ArgmaxCandidate warps[kArgmaxShardBlock / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warps[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        best = lane < kArgmaxShardBlock / 32 ? warps[lane] : argmax_shard_empty();
        best = argmax_shard_warp(best);
        if (lane == 0) {
            partial[static_cast<std::int64_t>(column) * tiles + blockIdx.x] = best;
        }
    }
}

// One CTA finishes every column. This is eight independent warp reductions at the actual
// T<=8 draft shapes, and a strided loop for larger public extents.
__device__ __forceinline__ void argmax_shard_finish_columns(
    const ArgmaxCandidate* partial, ArgmaxCandidate* output, int tiles, int columns) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (std::int64_t column = warp; column < columns; column += blockDim.x / 32) {
        ArgmaxCandidate best = argmax_shard_empty();
        for (int tile = lane; tile < tiles; tile += 32) {
            const ArgmaxCandidate value = partial[static_cast<std::int64_t>(column) * tiles + tile];
            if (argmax_shard_better(value, best)) { best = value; }
        }
        best = argmax_shard_warp(best);
        if (lane == 0) { output[column] = best; }
    }
}

__global__ __launch_bounds__(256) void argmax_shard_finish_kernel(
    const ArgmaxCandidate* partial, ArgmaxCandidate* output, int tiles, int columns) {
    argmax_shard_finish_columns(partial, output, tiles, columns);
}

__global__ __launch_bounds__(256) void argmax_shard_publish_kernel(
    const ArgmaxCandidate* partial, ArgmaxCandidate* payload, volatile std::uint32_t* flag,
    int tiles, int columns) {
    argmax_shard_finish_columns(partial, payload, tiles, columns);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        peer_mailbox_release(flag);
        __threadfence_system();
    }
}

// Only the receiver polls. Rank one's published slot is immutable until the entire graph
// retires, so it can proceed without an acknowledgment or a read of rank zero's storage.
__global__ __launch_bounds__(256) void argmax_shard_receive_kernel(
    const ArgmaxCandidate* partial, ArgmaxCandidate* local, const ArgmaxCandidate* peer,
    volatile std::uint32_t* peer_flag, volatile std::uint32_t* hang, int tiles, int columns,
    std::int32_t* out) {
    argmax_shard_finish_columns(partial, local, tiles, columns);
    __syncthreads();
    __shared__ std::uint32_t fault;
    if (threadIdx.x == 0) {
        peer_mailbox_wait(peer_flag, hang);
        fault = *hang;
    }
    __syncthreads();
    if (fault != 0) { return; }
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const ArgmaxCandidate a = local[column];
        const ArgmaxCandidate b = peer[column];
        out[column] = argmax_shard_better(b, a) ? b.index : a.index;
    }
}

__global__ __launch_bounds__(256) void argmax_shard_merge_kernel(
    const ArgmaxCandidate* local, const ArgmaxCandidate* peer, std::int32_t* out, int columns) {
    for (std::int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const ArgmaxCandidate a = local[column];
        const ArgmaxCandidate b = peer[column];
        out[column] = argmax_shard_better(b, a) ? b.index : a.index;
    }
}

} // namespace ninfer::ops::detail