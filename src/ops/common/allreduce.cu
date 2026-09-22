// Implements: include/ninfer/ops/allreduce.h
//
// Cross-device composition and exact local column interleave. Large eager no-P2P calls use
// caller-owned portable pinned
// buffers; each source D2H precedes inputs_ready and each peer H2D precedes pull_done. Other
// payloads/captures retain cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// The collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); optional local sum or column interleave
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ninfer/ops/peer_mailbox.h"
#include "core/layout.h"

#include "ops/kernel/peer_exchange.cuh" // detail::peer_exchange_sum_kernel
#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// MAILBOX TRANSPORT SELECTION. Returns the installed mailbox when every predicate holds:
//
//   * a PeerMailbox is installed for THIS ExecutionContext (Program setup created one);
//   * NINFER_TP2_MAILBOX is not "0"/"false" (the A/B escape hatch);
//   * the caller's stream is CAPTURING -- this collective is being recorded into a decode CUDA
//     graph, not issued eagerly (a prefill chunk, a warmup pass, or a --no-cuda-graph run),
//     because the mailbox's flag-reset protocol has a host reset point only between graph
//     replays, and eager calls would leave flags dirty for the next capture;
//   * the payload fits one slot and covers whole 16-byte vectors (staging beats zero-copy for
//     prefill-sized megabyte payloads; vector granularity is the exchange kernel's contract).
//
// Every predicate the mailbox fails is a collective that runs the staged path below, unchanged.
}  // namespace

namespace detail {
PeerMailbox* mailbox_transport(const ExecutionContext& ec, std::size_t bytes, cudaStream_t stream) {
    if (!PeerMailbox::enabled_by_environment()) { return nullptr; }
    PeerMailbox* mailbox = PeerMailbox::installed(ec);
    if (mailbox == nullptr) { return nullptr; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) { return nullptr; }
    if (status == cudaStreamCaptureStatusNone) { return nullptr; }
    if ((bytes % 16) != 0) { return nullptr; }
    if (bytes > mailbox->slot_bytes()) { return nullptr; }
    // The remaining predicate needs a slot: only a fresh, unclaimed slot makes this captured
    // call site exchange through the mailbox. A claim is permanent for the Program's lifetime,
    // which is exactly the captured-call-site identity the flag protocol needs; an exhausted
    // slab (null slot) is reported by the caller so the sizing mistake is visible.
    return mailbox;
}

}  // namespace detail

namespace {

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }

    int forward = 0;
    int reverse = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]));
    // Asymmetric support is not a usable transport for a symmetric collective: fall back to the
    // staged path rather than enabling one direction only.
    if (forward == 0 || reverse == 0) { return false; }

    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(pair[rank]);
        const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) {
            // Already enabled by an earlier call; clear the sticky runtime error so the next
            // cudaGetLastError() in an unrelated launcher does not observe it.
            cudaGetLastError();
            continue;
        }
        CUDA_CHECK(status);
    }
    return true;
}

PeerTransfer::PeerTransfer(const ExecutionContext& ec, std::size_t host_capacity_bytes) {
    require_two_devices(ec, "PeerTransfer: requires two distinct devices");
    const CurrentDeviceGuard guard;
    cudaEvent_t created[4] = {nullptr, nullptr, nullptr, nullptr};
    void* buffers[2] = {nullptr, nullptr};
    try {
        for (int rank = 0; rank < 2; ++rank) {
            devices_[rank] = ec.dev[rank]->device;
            streams_[rank] = ec.dev[rank]->stream;
            CurrentDeviceGuard::set(devices_[rank]);
            CUDA_CHECK(cudaEventCreateWithFlags(&created[rank], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&created[rank + 2], cudaEventDisableTiming));
            if (host_capacity_bytes != 0) {
                CUDA_CHECK(cudaHostAlloc(&buffers[rank], host_capacity_bytes, cudaHostAllocPortable));
            }
        }
    } catch (...) {
        for (void* buffer : buffers) {
            if (buffer != nullptr) { (void)cudaFreeHost(buffer); }
        }
        for (cudaEvent_t event : created) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
        throw;
    }
    inputs_ready_ = {created[0], created[1]};
    pull_done_ = {created[2], created[3]};
    host_buffers_ = {buffers[0], buffers[1]};
    host_capacity_ = host_capacity_bytes;
}

PeerTransfer::~PeerTransfer() {
    // Explicit host buffers cannot be freed until both readers retire, including unwinding after
    // a partially issued collective. Normal Program destruction already retires both streams.
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (host_capacity_ != 0) {
        for (int rank = 0; rank < 2; ++rank) {
            cudaError_t status = cudaSetDevice(devices_[rank]);
            if (status == cudaSuccess) { status = cudaStreamSynchronize(streams_[rank]); }
            if (status != cudaSuccess) {
                std::fprintf(stderr, "PeerTransfer stream retirement failed: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
        }
    }
    for (void*& buffer : host_buffers_) {
        if (buffer == nullptr) { continue; }
        const cudaError_t status = cudaFreeHost(buffer);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "PeerTransfer cudaFreeHost failed: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
        buffer = nullptr;
    }
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "PeerTransfer cudaEventDestroy failed: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
    if (previous >= 0) { (void)cudaSetDevice(previous); }
}

PeerTransfer::PeerTransfer(PeerTransfer&& other) noexcept
    : inputs_ready_(other.inputs_ready_), pull_done_(other.pull_done_),
      devices_(other.devices_), streams_(other.streams_), host_buffers_(other.host_buffers_),
      host_capacity_(other.host_capacity_) {
    other.inputs_ready_ = {nullptr, nullptr};
    other.pull_done_ = {nullptr, nullptr};
    other.devices_ = {-1, -1};
    other.streams_ = {nullptr, nullptr};
    other.host_buffers_ = {nullptr, nullptr};
    other.host_capacity_ = 0;
}

PeerTransfer& PeerTransfer::operator=(PeerTransfer&& other) noexcept {
    inputs_ready_.swap(other.inputs_ready_);
    pull_done_.swap(other.pull_done_);
    devices_.swap(other.devices_);
    streams_.swap(other.streams_);
    host_buffers_.swap(other.host_buffers_);
    std::swap(host_capacity_, other.host_capacity_);
    return *this;
}

bool PeerTransfer::matches(const ExecutionContext& ec) const noexcept {
    return live() && ec.tp == 2 && ec.dev[0] && ec.dev[1] &&
           devices_[0] == ec.dev[0]->device && devices_[1] == ec.dev[1]->device &&
           streams_[0] == ec.dev[0]->stream && streams_[1] == ec.dev[1]->stream;
}

bool PeerTransfer::uses_host_staging(const std::array<std::size_t, 2>& bytes) const {
    const std::size_t largest = std::max(bytes[0], bytes[1]);
    if (largest < minimum_host_staging_bytes || largest > host_capacity_) { return false; }
    require(live(), "PeerTransfer: moved-from transfer");
    const CurrentDeviceGuard guard;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(devices_[rank]);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(streams_[rank], &status));
        if (status != cudaStreamCaptureStatusNone) { return false; }
    }
    return true;
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerTransfer& transfer) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(transfer.matches(ec), "allreduce_sum: transfer must belong to this stream pair");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // MAILBOX TRANSPORT. When every predicate in mailbox_transport() holds, this collective
    // becomes one kernel per device: both ranks publish their partial into their own pinned host
    // slot, release a flag, spin on the peer's flag, and combine locally -- no events, no copy
    // engine, no driver round trip. Measured on this machine's transport-degraded pair (WDDM,
    // no P2P) this is ~41 us per 10 KiB reduction against the staged path's ~277 us, and the
    // two kernels' arithmetic is the same qualified combine as residual_add_launch below, so
    // the observable result is identical bit for bit.
    PeerMailbox* mailbox  = detail::mailbox_transport(ec, bytes, ec.dev[0]->stream);
    int slot              = -1;
    if (mailbox != nullptr) { slot = mailbox->take_capture_slot(); }
    if (slot >= 0) {
        const int vecs   = static_cast<int>(bytes / sizeof(detail::PeerVec));
        const int blocks = detail::peer_exchange_blocks(static_cast<int>(bytes));
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            detail::peer_exchange_sum_kernel<<<blocks, 256, 0, local.stream>>>(
                reinterpret_cast<detail::PeerVecBf16*>(buffer[rank].data),
                reinterpret_cast<detail::PeerVecBf16*>(buffer[rank].data),
                reinterpret_cast<detail::PeerVecBf16*>(mailbox->payload(rank, slot)),
                mailbox->flag(rank, slot),
                reinterpret_cast<const detail::PeerVecBf16*>(mailbox->payload(1 - rank, slot)),
                mailbox->flag(1 - rank, slot), mailbox->arrival(rank) + slot,
                mailbox->hang_word(), vecs);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    const bool host_staged = transfer.uses_host_staging({bytes, bytes});
    // Phase A: publish the completed input, or its completed portable pinned copy. The previous
    // call's phase C orders this D2H after the peer's last read of the same host buffer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        if (host_staged) {
            CUDA_CHECK(cudaMemcpyAsync(transfer.host_buffer(rank), buffer[rank].data, bytes,
                                       cudaMemcpyDeviceToHost, local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(rank), local.stream));
    }

    // Phase B: each rank pulls the peer's operand into storage only it owns. Both inbound copies
    // are issued before either rank waits again, so the two directions overlap.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.inputs_ready(1 - rank), 0));
        if (host_staged) {
            CUDA_CHECK(cudaMemcpyAsync(staging[rank].data, transfer.host_buffer(1 - rank), bytes,
                                       cudaMemcpyHostToDevice, local.stream));
        } else {
            CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.pull_done(rank), local.stream));
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerTransfer& transfer) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(transfer.matches(ec), "allgather_rows: transfer must belong to this stream pair");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    const bool host_staged = transfer.uses_host_staging({block[0], block[1]});
    // Publish both unequal source blocks before issuing either rank's peer wait.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        if (host_staged && block[rank] != 0) {
            CUDA_CHECK(cudaMemcpyAsync(transfer.host_buffer(rank), part[rank].data, block[rank],
                                       cudaMemcpyDeviceToHost, local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(rank), local.stream));
    }

    // Phase B: rank r writes its own block locally and pulls the peer's block, both on its own
    // stream, so destination[r] has exactly one writer.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                   part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                   local.stream));
        if (host_staged) {
            if (block[1 - rank] != 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    byte_offset(destination[rank].data, offset[1 - rank]),
                    transfer.host_buffer(1 - rank), block[1 - rank], cudaMemcpyHostToDevice,
                    local.stream));
            }
        } else {
            CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                                 part[1 - rank].data, block[1 - rank], local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.pull_done(rank), local.stream));
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.pull_done(1 - rank), 0));
    }
}

namespace {

// Each thread copies one scalar or 16-byte vector. All three pointers are resident on this
// launch's device: one source is local input, the other the completed packed peer pull.
// No BF16 conversion or arithmetic may enter this exact-storage path.
template <class Storage>
__global__ void interleave_columns_kernel(Storage* destination, const Storage* first,
                                           const Storage* second, int first_width,
                                           int second_width, int columns) {
    const int width = first_width + second_width;
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= static_cast<std::size_t>(width)) { return; }
    for (std::size_t column = blockIdx.y; column < static_cast<std::size_t>(columns);
         column += gridDim.y) {
        destination[column * width + row] = row < static_cast<std::size_t>(first_width)
            ? first[column * first_width + row]
            : second[column * second_width + row - first_width];
    }
}

void interleave_columns(const Tensor& destination, const Tensor& first, const Tensor& second,
                          cudaStream_t stream) {
    const int first_width = first.ne[0];
    const int second_width = second.ne[0];
    const auto addresses = reinterpret_cast<std::uintptr_t>(destination.data) |
                           reinterpret_cast<std::uintptr_t>(first.data) |
                           reinterpret_cast<std::uintptr_t>(second.data);
    constexpr int threads = 256;
    const auto height = static_cast<unsigned>(std::min(destination.ne[1], 65535));
    if ((first_width % 8) == 0 && (second_width % 8) == 0 && (addresses % 16) == 0) {
        const int width = (first_width + second_width) / 8;
        const dim3 grid(1 + (width - 1) / threads, height);
        interleave_columns_kernel<uint4><<<grid, threads, 0, stream>>>(
            static_cast<uint4*>(destination.data), static_cast<const uint4*>(first.data),
            static_cast<const uint4*>(second.data), first_width / 8, second_width / 8,
            destination.ne[1]);
    } else {
        const int width = first_width + second_width;
        const dim3 grid(1 + (width - 1) / threads, height);
        interleave_columns_kernel<std::uint16_t><<<grid, threads, 0, stream>>>(
            static_cast<std::uint16_t*>(destination.data),
            static_cast<const std::uint16_t*>(first.data),
            static_cast<const std::uint16_t*>(second.data), first_width, second_width,
            destination.ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t allgather_columns_workspace_capacity_bytes(std::int32_t peer_rows,
                                                        std::int32_t columns) {
    require(peer_rows > 0 && columns > 0,
            "allgather_columns: peer rows and columns must be positive");
    if (columns == 1) { return 0; }
    LayoutBuilder layout;
    (void)layout.add_tensor(DType::BF16, {peer_rows, columns}, 256,
                            "allgather columns packed peer");
    return layout.finish(256, "allgather columns workspace");
}

void allgather_columns(const std::array<Tensor, 2>& destination,
                       const std::array<Tensor, 2>& part,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& ec, const PeerTransfer& transfer) {
    require_two_devices(ec, "allgather_columns: two distinct devices are required");
    require(transfer.matches(ec), "allgather_columns: transfer must belong to this stream pair");
    const int columns = part[0].ne[1];
    const std::int64_t rows = static_cast<std::int64_t>(part[0].ne[0]) + part[1].ne[0];
    require(columns > 0 && rows <= std::numeric_limits<std::int32_t>::max(),
            "allgather_columns: invalid columns or combined row count");
    for (int rank = 0; rank < 2; ++rank) {
        require(part[rank].dtype == DType::BF16 && part[rank].ne[0] > 0 &&
                    part[rank].ne[1] == columns && part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_columns: parts must be BF16 [Vr,T] with matching T");
        require(destination[rank].dtype == DType::BF16 && destination[rank].ne[0] == rows &&
                    destination[rank].ne[1] == columns && destination[rank].ne[2] == 1 &&
                    destination[rank].ne[3] == 1,
                "allgather_columns: outputs must be BF16 [V0+V1,T]");
    }
    for (int rank = 0; rank < 2; ++rank) {
        require(part[rank].data && destination[rank].data && part[rank].is_contiguous() &&
                    destination[rank].is_contiguous(),
                "allgather_columns: storage must be non-null and contiguous");
#ifndef NDEBUG
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_columns: part is on the wrong device");
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_columns: output is on the wrong device");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         part[rank].bytes(), "allgather_columns: input overlaps output");
#endif
    }
    // For one token the two axes can be reinterpreted without any data movement or scratch.
    if (columns == 1) {
        const std::array<Tensor, 2> row_part{part[0].view({1, part[0].ne[0]}),
                                             part[1].view({1, part[1].ne[0]})};
        const std::array<Tensor, 2> row_destination{
            destination[0].view({1, static_cast<int>(rows)}),
            destination[1].view({1, static_cast<int>(rows)})};
        allgather_rows(row_destination, row_part, ec, transfer);
        return;
    }
    require(workspace[0] && workspace[1], "allgather_columns: rank-local arenas are required");
    auto scope0 = workspace[0]->scope();
    auto scope1 = workspace[1]->scope();
    std::array<Tensor, 2> received;
    const std::array<std::size_t, 2> bytes{part[0].bytes(), part[1].bytes()};
    for (int rank = 0; rank < 2; ++rank) {
        const auto capacity = allgather_columns_workspace_capacity_bytes(part[1 - rank].ne[0], columns);
        const DeviceSpan backing = workspace[rank]->alloc_bytes(capacity, 256);
        received[rank] = Tensor(backing.data, DType::BF16, {part[1 - rank].ne[0], columns});
#ifndef NDEBUG
        require_resident_on(backing.data, ec.dev[rank]->device,
                            "allgather_columns: workspace is on the wrong device");
        require_disjoint(backing.data, backing.bytes, part[rank].data, bytes[rank],
                         "allgather_columns: workspace overlaps input");
        require_disjoint(backing.data, backing.bytes, destination[rank].data,
                         destination[rank].bytes(), "allgather_columns: workspace overlaps output");
#endif
    }
    const CurrentDeviceGuard guard;
    const bool host_staged = transfer.uses_host_staging(bytes);
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        if (host_staged) {
            CUDA_CHECK(cudaMemcpyAsync(transfer.host_buffer(rank), part[rank].data, bytes[rank],
                                       cudaMemcpyDeviceToHost, local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.inputs_ready(1 - rank), 0));
        if (host_staged) {
            CUDA_CHECK(cudaMemcpyAsync(received[rank].data, transfer.host_buffer(1 - rank),
                                       bytes[1 - rank], cudaMemcpyHostToDevice, local.stream));
        } else {
            CUDA_CHECK(pull_peer(received[rank].data, part[1 - rank].data, bytes[1 - rank], local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.pull_done(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.pull_done(1 - rank), 0));
        interleave_columns(destination[rank], rank == 0 ? part[rank] : received[rank],
                            rank == 0 ? received[rank] : part[rank], local.stream);
    }
}


std::size_t gather_columns_to_rank0_workspace_capacity_bytes(std::int32_t peer_rows,
                                                              std::int32_t columns) {
    return allgather_columns_workspace_capacity_bytes(peer_rows, columns);
}

void gather_columns_to_rank0(Tensor& destination, const std::array<Tensor, 2>& part,
                             WorkspaceArena* rank0_workspace,
                             const ExecutionContext& ec, const PeerTransfer& transfer) {
    require_two_devices(ec, "gather_columns_to_rank0: two distinct devices are required");
    require(transfer.matches(ec), "gather_columns_to_rank0: transfer belongs to another stream pair");
    const int columns = part[0].ne[1];
    const std::int64_t rows = static_cast<std::int64_t>(part[0].ne[0]) + part[1].ne[0];
    require(columns > 0 && rows <= std::numeric_limits<std::int32_t>::max(),
            "gather_columns_to_rank0: invalid columns or combined row count");
    for (int rank = 0; rank < 2; ++rank) {
        const Tensor& source = part[rank];
        require(source.dtype == DType::BF16 && source.ne[0] > 0 && source.ne[1] == columns &&
                    source.ne[2] == 1 && source.ne[3] == 1 && source.data && source.is_contiguous(),
                "gather_columns_to_rank0: parts must be contiguous BF16 [Vr,T]");
#ifndef NDEBUG
        require_resident_on(source.data, ec.dev[rank]->device,
                            "gather_columns_to_rank0: part is on the wrong device");
#endif
    }
    require(destination.dtype == DType::BF16 && destination.ne[0] == rows &&
                destination.ne[1] == columns && destination.ne[2] == 1 &&
                destination.ne[3] == 1 && destination.data && destination.is_contiguous(),
            "gather_columns_to_rank0: output must be contiguous BF16 [V0+V1,T]");
#ifndef NDEBUG
    require_resident_on(destination.data, ec.dev[0]->device,
                        "gather_columns_to_rank0: output must belong to rank zero");
    require_disjoint(destination.data, destination.bytes(), part[0].data, part[0].bytes(),
                     "gather_columns_to_rank0: output overlaps local input");
#endif
    std::optional<WorkspaceArena::Scope> scope;
    Tensor received;
    if (columns > 1) {
        require(rank0_workspace != nullptr, "gather_columns_to_rank0: rank-zero arena is required");
        scope.emplace(rank0_workspace->scope());
        const auto capacity = gather_columns_to_rank0_workspace_capacity_bytes(part[1].ne[0], columns);
        const DeviceSpan backing = rank0_workspace->alloc_bytes(capacity, 256);
        received = Tensor(backing.data, DType::BF16, {part[1].ne[0], columns});
#ifndef NDEBUG
        require_resident_on(backing.data, ec.dev[0]->device,
                            "gather_columns_to_rank0: scratch must belong to rank zero");
        require_disjoint(backing.data, backing.bytes, part[0].data, part[0].bytes(),
                         "gather_columns_to_rank0: scratch overlaps local input");
        require_disjoint(backing.data, backing.bytes, destination.data, destination.bytes(),
                         "gather_columns_to_rank0: scratch overlaps output");
#endif
    } else {
        received = Tensor(byte_offset(destination.data, part[0].bytes()), DType::BF16,
                           {part[1].ne[0], 1});
    }

    const CurrentDeviceGuard guard;
    const bool staged = transfer.uses_host_staging({0, part[1].bytes()});
    const DeviceContext& source = *ec.dev[1];
    const DeviceContext& target = *ec.dev[0];
    CurrentDeviceGuard::set(source.device);
    if (staged) {
        CUDA_CHECK(cudaMemcpyAsync(transfer.host_buffer(1), part[1].data, part[1].bytes(),
                                   cudaMemcpyDeviceToHost, source.stream));
    }
    CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(1), source.stream));
    CurrentDeviceGuard::set(target.device);
    CUDA_CHECK(cudaStreamWaitEvent(target.stream, transfer.inputs_ready(1), 0));
    if (columns == 1) {
        CUDA_CHECK(cudaMemcpyAsync(destination.data, part[0].data, part[0].bytes(),
                                   cudaMemcpyDeviceToDevice, target.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(received.data, staged ? transfer.host_buffer(1) : part[1].data,
                               part[1].bytes(), staged ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,
                               target.stream));
    CUDA_CHECK(cudaEventRecord(transfer.pull_done(0), target.stream));
    if (columns > 1) { interleave_columns(destination, part[0], received, target.stream); }
    // The peer source and its optional pinned image are no longer needed after the pull,
    // even though rank zero's local interleave may still be running.
    CurrentDeviceGuard::set(source.device);
    CUDA_CHECK(cudaStreamWaitEvent(source.stream, transfer.pull_done(0), 0));
}
} // namespace ninfer::ops
