#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::uint32_t kPeerSpinLimit = 4000000u;

// The publishing writers must have fenced and met before one thread releases the flag.
// A multi-CTA publisher also waits for its per-device arrival counter before releasing.
__device__ __forceinline__ void peer_mailbox_release(volatile std::uint32_t* flag) {
    *flag = 1;
}

// One polling thread; its CTA rendezvous precedes payload reads. The owning Program checks
// the aggregate fault again at round retirement before consuming any output.
__device__ __forceinline__ void peer_mailbox_wait(volatile std::uint32_t* peer_flag,
                                                 volatile std::uint32_t* hang) {
    __threadfence_system();
    std::uint32_t spins = 0;
    while (*peer_flag != 1u) {
        __nanosleep(100);
        if (++spins > kPeerSpinLimit) {
            *hang = 1;
            break;
        }
    }
    __threadfence_system();
}

} // namespace ninfer::ops::detail