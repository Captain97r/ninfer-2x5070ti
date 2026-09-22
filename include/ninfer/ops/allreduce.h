#pragma once

// Two-device collectives with one explicitly owned PeerTransfer per stream pair.
//
// Transport preserves every source byte. A Program without direct P2P can provision two portable
// pinned buffers at setup. Large eager collectives use explicit D2H followed by H2D on each
// destination stream. Captured calls and payloads outside the provisioned range retain the
// existing mailbox or CUDA cross-device-copy path; no allocation occurs inside an Op.
//
// Whole-buffer staging and the CUDA cross-device-copy path issue three phases for BOTH ranks:
//   A: [optional D2H to this rank's pinned buffer], record(inputs_ready[r])
//   B: wait(inputs_ready[1-r]), pull peer input, record(pull_done[r])
//   C: wait(pull_done[1-r]), [local sum for allreduce_sum]
// For explicit staging the pull is H2D from the peer's pinned buffer. Otherwise it is the
// existing cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over UVA pointers. The final wait protects
// source AND pinned-buffer reuse: the next call's D2H cannot overwrite a buffer the peer still
// reads. Consecutive calls therefore need no host synchronization.
//
// The qualified automatic 10 MiB eager allreduce uses two 5 MiB tiles. Each source's owned D2H
// stream waits for inputs_ready on its main stream, publishes two tile-ready events, and each
// peer main stream pulls those tiles in order. Both pull_done records precede either final
// peer wait. Each main stream joins its own D2H completion and the peer's pull before the same
// full-buffer local sum. Thus the main streams protect input, scratch and pinned-buffer reuse
// across tiled, whole-buffer, gathered and captured calls without a host wait.
//
// Work completes in DeviceContext::stream order, including any owned auxiliary D2H work. These
// nonblocking streams do not implicitly wait for legacy-default-stream initialization: callers
// must retire such initialization before use.
// Return means enqueued, not completed; CUDA Async APIs may still block in the driver.
// The caller owns the stream pair and resource, and destroys graph users before the resource.

#include "core/arena.h"
#include "core/device.h" // DeviceContext, ExecutionContext
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaEvent_t

#include <array>
#include <cstddef>

namespace ninfer::ops {

class PeerMailbox;  // see ops/peer_mailbox.h; forward-declared to keep this header's includes as-is

// Probes cudaDeviceCanAccessPeer in both directions and enables peer access on both devices only
// when both directions report support; a device that already had peer access enabled is left
// alone. Returns true when direct P2P is active for the pair, false when the driver denies it and
// the collectives below can use explicit or CUDA-managed host staging. Never fails on
// denial: the staged path is a supported transport, not an error.
//
// Call once during setup. cudaDeviceEnablePeerAccess is a context-level operation, not
// stream-ordered and not graph-capturable; it must never appear in a hot path.
bool enable_peer_access(const ExecutionContext& ec);

namespace detail {
// Chooses the mailbox transport for this collective when every predicate holds: environment
// override on, a mailbox installed for this exact device pair, THIS call inside a stream capture,
// and the payload within a slot. Returns the live mailbox, or null for the staged path. Defined
// in allreduce.cu; the pointer is mutable because claiming a capture slot is a mutating claim.
[[nodiscard]] PeerMailbox* mailbox_transport(const ExecutionContext& ec, std::size_t bytes,
                                             cudaStream_t stream);
}  // namespace detail

// Internal closed schedule selection. Automatic provisions the two-tile allreduce only for the
// measured Windows WDDM sm_120/70-SM pair with no direct P2P in either direction. WholeBuffer
// retains the otherwise identical transport as a reference; it is not a product option.
enum class PeerTransferSchedule { Automatic, WholeBuffer };

// One Program/stream-pair resource: four ordering events, optional pinned staging and, for the
// qualified automatic route, one nonblocking D2H stream plus two tile-ready events per rank.
// Provisioned bytes are PER RANK. Pass zero for direct P2P or implicit-only comparison.
// No buffer is shared between owners. The constructor allocates outside capture; the destructor
// retires both bound streams and any owned D2H streams before releasing resources, including
// after a partially enqueued call. Bound streams and input/scratch allocations must outlive this
// object's pending work; graph executables using its events must be destroyed before it.
// Moves transfer ownership, including pending work and the original stream-pair binding.
// Move assignment exchanges both resources; each object then owns the other's former work.
class PeerTransfer {
public:
    static constexpr std::size_t minimum_host_staging_bytes = 64u * 1024u;
    static constexpr std::size_t two_tile_allreduce_bytes = 10u << 20;

    explicit PeerTransfer(const ExecutionContext& ec, std::size_t host_capacity_bytes = 0,
                          PeerTransferSchedule schedule = PeerTransferSchedule::Automatic);
    ~PeerTransfer();

    PeerTransfer(const PeerTransfer&) = delete;
    PeerTransfer& operator=(const PeerTransfer&) = delete;
    PeerTransfer(PeerTransfer&& other) noexcept;
    PeerTransfer& operator=(PeerTransfer&& other) noexcept;

    [[nodiscard]] cudaEvent_t inputs_ready(int rank) const noexcept {
        return inputs_ready_[static_cast<std::size_t>(rank)];
    }
    [[nodiscard]] cudaEvent_t pull_done(int rank) const noexcept {
        return pull_done_[static_cast<std::size_t>(rank)];
    }
    [[nodiscard]] bool live() const noexcept {
        return inputs_ready_[0] != nullptr && inputs_ready_[1] != nullptr &&
               pull_done_[0] != nullptr && pull_done_[1] != nullptr;
    }
    [[nodiscard]] bool matches(const ExecutionContext& ec) const noexcept;
    // Checks size before querying capture status, so small decode calls pay no CUDA query.
    // Both stream statuses must be None; a captured fallback never touches pinned staging.
    [[nodiscard]] bool uses_host_staging(const std::array<std::size_t, 2>& bytes) const;
    // Complete call-time admission: setup qualified this pair, exactly 10 MiB, sufficient
    // capacity, and neither bound stream is capturing. Other sizes pay no capture query.
    [[nodiscard]] bool uses_two_tile_allreduce(std::size_t bytes) const;
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept { return host_capacity_; }
    [[nodiscard]] void* host_buffer(int rank) const noexcept {
        return host_buffers_[static_cast<std::size_t>(rank)];
    }

private:
    friend void allreduce_sum(const std::array<Tensor, 2>& buffer,
                              const std::array<Tensor, 2>& staging,
                              const ExecutionContext& ec, const PeerTransfer& transfer);

    std::array<cudaEvent_t, 2> inputs_ready_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> pull_done_{nullptr, nullptr};
    std::array<cudaStream_t, 2> d2h_streams_{nullptr, nullptr};
    std::array<std::array<cudaEvent_t, 2>, 2> tile_ready_{};
    std::array<int, 2> devices_{-1, -1};
    std::array<cudaStream_t, 2> streams_{nullptr, nullptr};
    std::array<void*, 2> host_buffers_{nullptr, nullptr};
    std::size_t host_capacity_ = 0;
};

/**
 * Two-device summing all-reduce, in place:
 *
 *   ideal[i] = buffer_rank0[i] + buffer_rank1[i]   for every i,
 *
 * written back to both `buffer[0]` and `buffer[1]`, which afterwards hold the identical sum.
 *
 * `buffer[r]` is a contiguous BF16 tensor resident on `ec.dev[r]`; both ranks carry the same
 * shape. `staging[r]` is scratch of the same dtype and shape, also resident on `ec.dev[r]`, and
 * must not overlap `buffer[r]`; it receives the peer's contribution and its contents after the
 * call are unspecified. Rank r's stream is the only stream that ever touches `staging[r]`.
 *
 * The oracle evaluates `ideal` in FP64 from the represented BF16 inputs; the observable BF16
 * output is that value rounded once to BF16 storage, and the local combine is the qualified
 * residual_add computation body (FP32 accumulation of the two BF16 operands, one
 * round-to-nearest-even on store). The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `transfer`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerTransfer& transfer);

/**
 * Two-device row gather, exact (no arithmetic). The gathered axis is `ne[1]`, so each rank
 * contributes one contiguous block of a `[C, R]` tensor (`ne[0] == C` is the row length,
 * `ne[1] == R` the row count, `ne[2] == ne[3] == 1`):
 *
 *   destination[r][c, row] = part[0][c, row]                    for row <  part[0].ne[1]
 *   destination[r][c, row] = part[1][c, row - part[0].ne[1]]    otherwise
 *
 * for both r, so each device ends up holding the identical full image. `part[r]` is the row range
 * owned by `ec.dev[r]`: rank 0 owns the leading rows and rank 1 the trailing rows, and the two
 * counts must sum to the destination row count. `destination[r]` and `part[r]` are contiguous,
 * share one dtype, agree on `ne[0]`, live on `ec.dev[r]`, and must not overlap. A caller whose
 * split axis is not `ne[1]` (for example `[V, B]` logits split by vocabulary with B > 1) arranges
 * the layout so that it is, or uses allgather_columns for BF16 inputs. This Op relocates
 * contiguous storage and performs no transpose.
 *
 * `destination[r]` is written only by rank r's stream: rank r copies its own block locally and
 * pulls the peer's block. The transformation is a pure relocation of storage, so it is verified by
 * exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `transfer`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerTransfer& transfer);

// Per-rank scratch bound for allgather_columns: pass the OTHER rank's physical row count.
// A single column is already contiguous and needs no scratch. Alignment is included.
[[nodiscard]] std::size_t allgather_columns_workspace_capacity_bytes(
    std::int32_t peer_rows, std::int32_t columns);

/**
 * Two-device BF16 column gather, exact storage relocation along ne[0]:
 *
 *   destination[r][v,t] = part[0][v,t]         for v < V0,
 *                          part[1][v-V0,t]    otherwise.
 *
 * Each part[r] is contiguous BF16 [Vr,T], Vr>0 and T>0, resident on ec.dev[r]. Both outputs
 * are contiguous BF16 [V0+V1,T] on their respective devices, and V0+V1 fits int32. Every bit is
 * preserved, including NaN payloads, signed zero and padding rows; there is no valid-row filter.
 * Inputs are unchanged. Input, output and workspace storage on each rank must be disjoint.
 *
 * transfer belongs to these two distinct devices and streams. For T>1, workspace[r] supplies
 * the bound queried with V(1-r),T; only caller-owned arenas are suballocated, and their scopes
 * are restored on return. For T==1 the arena pointers may be null. No device allocation,
 * persistent state or host synchronization is introduced. Peer reads use the transfer's pinned
 * eager or UVA fallback path; local kernels never dereference the other GPU's allocation.
 *
 * All work uses ec's streams. On return both streams are ordered after the peer's source read
 * and their own output/scratch use, so inputs and arenas may be reused by subsequent calls on
 * the same stream pair without a host wait. Captured addresses remain caller-owned and stable.
 */
void allgather_columns(const std::array<Tensor, 2>& destination,
                       const std::array<Tensor, 2>& part,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& ec, const PeerTransfer& transfer);

// Rank-zero scratch bound for gather_columns_to_rank0. Rank one needs no scratch.
// peer_rows is rank one's physical row count; T1 returns zero.
[[nodiscard]] std::size_t gather_columns_to_rank0_workspace_capacity_bytes(
    std::int32_t peer_rows, std::int32_t columns);

/**
 * Two-device BF16 column gather to rank zero, exact storage relocation:
 *
 *   destination[v,t] = part[0][v,t]       for v < V0,
 *                      part[1][v-V0,t]   otherwise.
 *
 * Parts are contiguous BF16 [Vr,T] on ec.dev[r], Vr>0, T>0, V0+V1 fits int32.
 * The one destination is contiguous BF16 [V0+V1,T] on ec.dev[0]. Every source bit,
 * including NaN payloads and vocabulary padding, is preserved. Both parts are unchanged.
 * Rank zero's input, output and scratch must be disjoint; no peer pointer is read by a kernel.
 *
 * rank0_workspace supplies the queried aligned capacity for T>1; its cursor is restored on
 * return. It may be null at T1. There is no allocation, persistent state or host wait.
 * transfer belongs to ec's stream pair. The peer source uses explicit eager staging when
 * provisioned and eligible, otherwise a CUDA UVA copy. Work is enqueued on owning streams;
 * rank one's stream is ordered after its source/pinned-buffer read before subsequent reuse.
 * Rank zero's stream orders the final output and scratch use. Captured storage remains owned
 * and address-stable until its graph users retire.
 */
void gather_columns_to_rank0(Tensor& destination, const std::array<Tensor, 2>& part,
                             WorkspaceArena* rank0_workspace,
                             const ExecutionContext& ec, const PeerTransfer& transfer);
} // namespace ninfer::ops
