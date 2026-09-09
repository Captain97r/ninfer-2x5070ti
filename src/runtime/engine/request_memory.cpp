#include "runtime/engine/request_memory.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

class RequestMemory::Impl {
public:
    Impl(DeviceContext& context, std::size_t capacity) : device(context.device) {
        if (capacity != 0) {
            // The arena can be built while a DIFFERENT device is current (a tp2 instance builds
            // rank 1's request memory from rank 0's setup path), and the previous device must
            // survive it: the caller's own subsequent allocations (the rank-0 program's
            // persistent span among them) would otherwise land on this device, and every later
            // use of them on their intended device fails with cudaErrorInvalidValue.
            int previous = 0;
            CUDA_CHECK(cudaGetDevice(&previous));
            CUDA_CHECK(cudaSetDevice(device));
            try {
                arena = std::make_unique<DeviceArena>(capacity);
            } catch (...) {
                (void)cudaSetDevice(previous);
                throw;
            }
            CUDA_CHECK(cudaSetDevice(previous));
        }
    }

    ~Impl() {
        if (arena != nullptr) {
            int previous = device;
            CUDA_CHECK(cudaGetDevice(&previous));
            if (previous != device) { CUDA_CHECK(cudaSetDevice(device)); }
            arena.reset();
            if (previous != device) { (void)cudaSetDevice(previous); }
        }
    }

    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    std::size_t active_bytes     = 0;
    std::size_t active_alignment = 1;
    std::size_t peak_bytes       = 0;
};

RequestMemory::RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes)
    : impl_(std::make_unique<Impl>(device, frozen_capacity_bytes)) {}

RequestMemory::~RequestMemory() = default;

void RequestMemory::activate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        if (alignment != 1) {
            throw std::invalid_argument("an empty transient region must use alignment one");
        }
        deactivate();
        return;
    }

    if (!is_power_of_two(alignment) || alignment > kDeviceAllocationAlignment) {
        throw std::invalid_argument("unsupported transient region alignment");
    }

    if (impl_->arena == nullptr || bytes > impl_->arena->capacity()) {
        throw std::invalid_argument("request transient exceeds its frozen startup capacity");
    }

    impl_->active_bytes     = bytes;
    impl_->active_alignment = alignment;
    impl_->peak_bytes       = std::max(impl_->peak_bytes, bytes);
}

void RequestMemory::deactivate() noexcept {
    impl_->active_bytes     = 0;
    impl_->active_alignment = 1;
}

TransientRegion RequestMemory::region() const noexcept {
    if (impl_->active_bytes == 0) { return {}; }
    return {static_cast<std::byte*>(impl_->arena->base()), impl_->active_bytes,
            impl_->active_alignment};
}

ArenaMemorySummary RequestMemory::summary() const noexcept {
    return ArenaMemorySummary{
        impl_->arena != nullptr ? impl_->arena->capacity() : 0,
        impl_->active_bytes,
        impl_->peak_bytes,
    };
}

void RequestMemory::reset_peak() noexcept { impl_->peak_bytes = impl_->active_bytes; }

} // namespace ninfer::runtime
