#pragma once

#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to speculative_prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Independently for each row b, greedy mode accepts the longest available draft prefix matching
 *   target_tokens and commits the target token at the first mismatch (or the bonus column).
 *   Sampling mode applies configs[b] to each valid verification column, accepts draft i with
 *   target probability p_i(draft_i), samples from the residual distribution on first rejection,
 *   and samples a bonus from column Pcur[b] when every available draft is accepted. The draft
 *   proposal distribution is one-hot at each greedy draft token.
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B], drafts is
 *   I32 [K,B], logits is BF16 [physical_rows,K+1,B], and current_extents/lengths/anchors/
 *   licensed_counts/accepted are I32 [B]. token_domain is in [1,physical_rows], K>=1, B>=1, and
 *   configs points to a device-resident SamplingConfig[B]. Tensor arguments, configs, and
 *   configs[b].token_counts do not overlap except for the explicitly mutated objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   For each row, let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives
 *   accepted drafts, licensed_tokens[A,b] receives the correction/bonus token, and the remaining
 *   physical slots are zero. licensed_counts[b]=L; accepted[b]=A; anchors[b] becomes the
 *   correction/bonus token; lengths[b]+=L. In sampling mode, each produced token increments
 *   configs[b].token_counts when that pointer is non-null. Greedy mode does not update
 *   token_counts. current_extents and all other inputs remain unchanged. Request statistics are
 *   deliberately outside this Op.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes().
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);


// A complete acceptance result. Views are rank-local and non-owning; scalar fields are I32[B],
// licensed_tokens is I32[K+1,B]. No pointer-bearing sampling config is part of this decision.
struct SpeculativeDecisionView {
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
};

// Per-rank aligned transient bound, including the compact record's transport padding.
// The admitted decision capacity is 1<=K<=5 and 1<=B<=8.
[[nodiscard]] std::size_t speculative_replicate_decision_workspace_capacity_bytes(
    std::int32_t drafts, std::int32_t batch);

/**
 * Copy one completed acceptance decision from rank zero and apply its exact state on rank one.
 * For each row, copies every licensed_tokens slot and the frontier, anchor, licensed count and
 * accepted count. The rank-zero views are unchanged. The source is a valid acceptance result:
 * A in [0,K], L=A+1, tokens[0:L] are valid token IDs, tokens[L:K+1] are zero, and its scalar
 * fields already reflect acceptance. It is copied verbatim, not recomputed from rank one's state.
 *
 * For a peer config with temperature>0 and non-null token_counts, increment the local counter
 * for each licensed token exactly once, including repeated tokens. Otherwise counters are
 * unchanged. This is the same update as speculative_accept_greedy_drafts; the peer must not
 * also run acceptance for this decision. Configs and all other counter entries are unchanged.
 * No sampling, floating-point arithmetic, RNG or host transaction policy is performed.
 *
 * Both views are contiguous I32 with the shapes above on ec.dev[r]. Fields, local workspace,
 * peer_configs and its counter allocations do not overlap. peer_configs is SamplingConfig[B]
 * resident on rank one, and each non-null counter pointer names that rank's valid token domain.
 * K and B fit the declared capacity. The caller supplies both queried arena bounds, the two
 * streams, and their PeerTransfer. Arena cursors are restored; there is no hidden allocation.
 *
 * Captured calls may publish into the existing immutable-per-round mailbox slot; its owner
 * must reset flags only after both streams retire and validate_completed_round before consuming
 * output. Otherwise an event-ordered compact CUDA copy protects source scratch reuse. Kernel
 * consumers access only local GPU storage or the supported mapped host mailbox. Eager and
 * captured consecutive calls require no intervening host synchronization.
 */
void speculative_replicate_decision(
    const std::array<SpeculativeDecisionView, 2>& decision, const SamplingConfig* peer_configs,
    const std::array<WorkspaceArena*, 2>& workspace,
    const ExecutionContext& execution, const PeerTransfer& transfer);
} // namespace ninfer::ops
