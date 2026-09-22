#pragma once

#include "ninfer/ops/sampling.h"
#include "ops/kernel/peer_mailbox_device.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

struct SpeculativeDecisionPointers {
    std::int32_t* frontiers;
    std::int32_t* anchors;
    std::int32_t* licensed_tokens;
    std::int32_t* licensed_counts;
    std::int32_t* accepted_drafts;
};

template <bool Publish>
__global__ __launch_bounds__(256) void speculative_decision_pack_kernel(
    SpeculativeDecisionPointers source, std::int32_t* packed, int width, int batch,
    int transfer_words, volatile std::uint32_t* flag) {
    const int index = static_cast<int>(threadIdx.x);
    const int stride = width + 4;
    if (index < transfer_words) {
        std::int32_t value = 0;
        if (index < stride * batch) {
            const int row = index / stride;
            const int field = index % stride;
            if (field < width) { value = source.licensed_tokens[row * width + field]; }
            else if (field == width) { value = source.frontiers[row]; }
            else if (field == width + 1) { value = source.anchors[row]; }
            else if (field == width + 2) { value = source.licensed_counts[row]; }
            else { value = source.accepted_drafts[row]; }
        }
        packed[index] = value;
    }
    if constexpr (Publish) {
        // Every payload writer fences before the publishing thread releases this call-site slot.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            peer_mailbox_release(flag);
            __threadfence_system();
        }
    }
}

template <bool Mailbox>
__global__ __launch_bounds__(256) void speculative_decision_apply_kernel(
    const std::int32_t* packed, SpeculativeDecisionPointers destination,
    const SamplingConfig* configs, int width, int batch,
    volatile std::uint32_t* flag, volatile std::uint32_t* hang) {
    if constexpr (Mailbox) {
        __shared__ std::uint32_t fault;
        if (threadIdx.x == 0) {
            peer_mailbox_wait(flag, hang);
            fault = *hang;
        }
        __syncthreads();
        if (fault != 0) { return; }
    }
    const int row = static_cast<int>(threadIdx.x);
    if (row >= batch) { return; }
    const std::int32_t* record = packed + row * (width + 4);
    for (int column = 0; column < width; ++column) {
        destination.licensed_tokens[row * width + column] = record[column];
    }
    destination.frontiers[row] = record[width];
    destination.anchors[row] = record[width + 1];
    const int produced = record[width + 2];
    destination.licensed_counts[row] = produced;
    destination.accepted_drafts[row] = record[width + 3];
    const SamplingConfig config = configs[row];
    if (config.temperature > 0.0f && config.token_counts != nullptr) {
        for (int column = 0; column < produced; ++column) {
            atomicAdd(&config.token_counts[record[column]], 1);
        }
    }
}

} // namespace ninfer::ops::detail
