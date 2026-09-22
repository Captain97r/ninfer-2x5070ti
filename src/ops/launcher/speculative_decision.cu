// Implements the exact state replication declared in ninfer/ops/speculative_round.h.
#include "ops/launcher/speculative_decision.h"

#include "ninfer/ops/peer_mailbox.h"
#include "ops/kernel/speculative_decision.cuh"

namespace ninfer::ops::detail {
namespace {

SpeculativeDecisionPointers pointers(const SpeculativeDecisionView& view) {
    return {static_cast<std::int32_t*>(view.frontiers.data),
            static_cast<std::int32_t*>(view.anchors.data),
            static_cast<std::int32_t*>(view.licensed_tokens.data),
            static_cast<std::int32_t*>(view.licensed_counts.data),
            static_cast<std::int32_t*>(view.accepted_drafts.data)};
}

class DeviceRestore {
public:
    DeviceRestore() { CUDA_CHECK(cudaGetDevice(&previous_)); }
    ~DeviceRestore() { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

} // namespace

void speculative_replicate_decision_launch(
    const std::array<SpeculativeDecisionView, 2>& decision, const SamplingConfig* peer_configs,
    const std::array<DeviceSpan, 2>& scratch, std::size_t transfer_bytes,
    const ExecutionContext& execution, const PeerTransfer& transfer) {
    const DeviceRestore restore;
    const int width = decision[0].licensed_tokens.ne[0];
    const int batch = decision[0].licensed_tokens.ne[1];
    const int words = static_cast<int>(transfer_bytes / sizeof(std::int32_t));
    const auto source = pointers(decision[0]);
    const auto destination = pointers(decision[1]);
    PeerMailbox* mailbox = mailbox_transport(execution, transfer_bytes, execution.dev[0]->stream);
    const int slot = mailbox != nullptr ? mailbox->take_capture_slot() : -1;
    if (slot >= 0) {
        // The rank-zero slot is immutable until the entire graph retires. Rank zero therefore
        // does not wait for acknowledgment, and may reuse its own workspace after publishing.
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        speculative_decision_pack_kernel<true><<<1, 256, 0, execution.dev[0]->stream>>>(
            source, static_cast<std::int32_t*>(mailbox->payload(0, slot)), width, batch, words,
            mailbox->flag(0, slot));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        speculative_decision_apply_kernel<true><<<1, 256, 0, execution.dev[1]->stream>>>(
            static_cast<const std::int32_t*>(mailbox->payload(0, slot)), destination, peer_configs,
            width, batch, mailbox->flag(0, slot), mailbox->hang_word());
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // One-way event-ordered pull. Both record calls precede the waits that snapshot them.
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    speculative_decision_pack_kernel<false><<<1, 256, 0, execution.dev[0]->stream>>>(
        source, static_cast<std::int32_t*>(scratch[0].data), width, batch, words, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(0), execution.dev[0]->stream));
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.dev[1]->stream, transfer.inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(scratch[1].data, scratch[0].data, transfer_bytes,
                               cudaMemcpyDeviceToDevice, execution.dev[1]->stream));
    CUDA_CHECK(cudaEventRecord(transfer.pull_done(1), execution.dev[1]->stream));
    speculative_decision_apply_kernel<false><<<1, 256, 0, execution.dev[1]->stream>>>(
        static_cast<const std::int32_t*>(scratch[1].data), destination, peer_configs,
        width, batch, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.dev[0]->stream, transfer.pull_done(1), 0));
}

} // namespace ninfer::ops::detail
