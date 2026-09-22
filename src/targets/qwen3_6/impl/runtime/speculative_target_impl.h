#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, execution.work, execution.device.stream);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer, ops::GqaExecutionEnvelope envelope) {
    if (execution.peer == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    if (frame.feature_sink != nullptr || peer.feature_sink != nullptr) {
        // DFlash is the only feature-sink user and it is rejected at tp2 before reaching here.
        throw std::logic_error("tensor-parallel target verify has no feature-sink path");
    }
    // Complete-pipeline measurements favor rank-zero acceptance only in captured execution.
    // Select once and pass the same decision through the head and acceptance schedule. Wider
    // reference domains keep replicated acceptance; the compact decision Op admits MTP K1..5.
    const CurrentDevice restore;
    CUDA_CHECK(cudaSetDevice(execution.device.device));
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(execution.device.stream, &capture_status));
    const TpTargetHead head =
        capture_status == cudaStreamCaptureStatusActive && frame.drafts.ne[0] >= 1 &&
                frame.drafts.ne[0] <= static_cast<std::int32_t>(kMtpDecodeMaximumDrafts)
            ? TpTargetHead::RankZero
            : TpTargetHead::Replicated;
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.target_verify_batch({frame.ids, peer.ids},
                             {frame.cache_positions, peer.cache_positions},
                             {frame.rope_positions, peer.rope_positions},
                             {frame.valid_columns, peer.valid_columns},
                             {frame.kv_table_rows, peer.kv_table_rows}, {frame.lanes, peer.lanes},
                             envelope, {frame.target_hidden, peer.target_hidden},
                             {frame.target_logits, peer.target_logits},
                             {frame.target_tokens, peer.target_tokens}, head);
    const ExecutionContext& ec = *execution.peer->execution;
    const std::array<WorkspaceArena*, 2> work{&execution.work, execution.peer->work};
    const std::array<TargetVerifyFrameView*, 2> views{&frame, &peer};
    const auto accept = [&](int rank) {
        const auto r             = static_cast<std::size_t>(rank);
        TargetVerifyFrameView& v = *views[r];
        ops::speculative_accept_greedy_drafts(v.target_tokens, v.target_logits, v.drafts,
                                              v.current_extents, v.frontiers, v.anchors,
                                              v.licensed_tokens, v.licensed_counts,
                                              v.accepted_drafts, TextConfig::token_domain,
                                              v.sampling, *work[r], ec.dev[rank]->stream);
    };
    const auto select_hidden = [&](int rank) {
        TargetVerifyFrameView& v = *views[static_cast<std::size_t>(rank)];
        ops::speculative_select_accepted_hidden(v.target_hidden, v.accepted_drafts,
                                                v.selected_hidden, ec.dev[rank]->stream);
    };
    if (head == TpTargetHead::RankZero) {
        accept(0);
        const auto decision_view = [](const TargetVerifyFrameView& v) {
            return ops::SpeculativeDecisionView{v.frontiers, v.anchors, v.licensed_tokens,
                                                 v.licensed_counts, v.accepted_drafts};
        };
        ops::speculative_replicate_decision({decision_view(frame), decision_view(peer)},
                                             peer.sampling, work, ec, *execution.peer->transfer);
        // Peer decision and stochastic counter increments precede hidden selection, next-round
        // controls and alignment on that same stream. The host commit remains rank-zero owned.
        for_each_rank(ec, select_hidden);
    } else {
        // Preserve eager issue order as well as its replicated arithmetic and counter updates.
        for_each_rank(ec, [&](int rank) {
            accept(rank);
            select_hidden(rank);
        });
    }
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
