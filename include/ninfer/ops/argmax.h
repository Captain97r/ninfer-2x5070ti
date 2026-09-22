#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

class PeerTransfer;

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * logits is contiguous BF16 [physical_rows,T], out is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. out must not overlap logits.
 * If row zero is NaN, its index is returned; other NaNs are ignored. Infinities and signed
 * zeros otherwise follow the same value comparison and lowest-index tie rule.
 * The Op has no workspace and changes no state other than writing all of out.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// Per-rank scratch bound, shared by mailbox and staged transport. Zero columns need no scratch.
[[nodiscard]] std::size_t argmax_row_parallel_workspace_capacity_bytes(
    std::int32_t local_physical_rows, std::int32_t columns);

/**
 * Exact argmax of two vocabulary-row shards, returned on execution rank zero:
 *
 *   logical[v,t] = logits[0][v,t]                when v < V0,
 *                  logits[1][v - V0,t]          otherwise;
 *   out[t] = argmax(logical[:,t], valid_rows), using the contract above.
 *
 * Each logits[r] is contiguous BF16 [Vr,T], resident on execution.dev[r], with Vr>0 and T>=0.
 * V0+V1 fits int32 and 1 <= valid_rows <= V0+V1. Padding beyond valid_rows is never considered;
 * rank one's valid range may be empty. out is contiguous I32 [T] on execution rank zero.
 * Inputs, output and each rank's workspace are disjoint. Inputs are unchanged.
 *
 * execution contains two distinct devices; transfer belongs to their stream pair. The Op uses
 * those compute streams and suballocates only from the supplied rank-local arenas. No device
 * allocation or host synchronization occurs. On return each stream is ordered after every use
 * of its scratch, including the peer's staged read, so successive calls can reuse the arenas.
 * A captured mailbox call claims one slot for its owning Program's lifetime. The caller retires
 * both devices and validates its fault word before consuming output or resetting replay flags.
 * The only semantic output is out on rank zero; proposal token-ID mapping is not performed here.
 */
void argmax_row_parallel(const std::array<Tensor, 2>& logits, Tensor& out,
                         std::int32_t valid_rows,
                         const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& execution, const PeerTransfer& transfer);

} // namespace ninfer::ops
