#include "ops/launcher/argmax_row_parallel.h"
#include "ops/kernel/argmax_row_parallel.cuh"
#include "ops/common/token_slices.h"
#include "ninfer/ops/peer_mailbox.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::ops::detail {
namespace {

class DeviceRestore {
public:
    DeviceRestore() { CUDA_CHECK(cudaGetDevice(&previous_)); }
    ~DeviceRestore() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "argmax device cleanup failed: %s\n", cudaGetErrorString(status));
        }
    }
private:
    int previous_ = 0;
};

} // namespace

void argmax_row_parallel_launch(const std::array<Tensor, 2>& logits, Tensor& out,
                                std::int32_t valid_rows,
                                const std::array<ArgmaxRowParallelWorkspace, 2>& workspace,
                                const ExecutionContext& execution, const PeerEvents& events) {
    const DeviceRestore restore;
    const int columns = logits[0].ne[1];
    const std::size_t bytes = sizeof(ArgmaxCandidate) * static_cast<std::size_t>(columns);
    PeerMailbox* mailbox = mailbox_transport(execution, bytes, execution.dev[0]->stream);
    const int slot = mailbox != nullptr ? mailbox->take_capture_slot() : -1;

    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        const int offset = rank == 0 ? 0 : logits[0].ne[0];
        const int valid = std::clamp(valid_rows - offset, 0, logits[rank].ne[0]);
        const auto& w = workspace[rank];
        for_each_token_slice(columns, 1, [&](int begin, int count) {
            const auto* input = static_cast<const __nv_bfloat16*>(logits[rank].data) +
                                static_cast<std::int64_t>(begin) * logits[rank].ne[0];
            auto* partial = w.partial + static_cast<std::int64_t>(begin) * w.tiles;
            argmax_shard_partial_kernel<<<dim3(w.tiles, count), kArgmaxShardBlock, 0,
                                            execution.dev[rank]->stream>>>(
                input, partial, logits[rank].ne[0], valid, offset, w.tiles);
            CUDA_CHECK(cudaGetLastError());
        });
    }

    if (slot >= 0) {
        // Queue the producer first. Both streams are enrolled in the same capture; the
        // receiver's system-memory flag supplies the data dependency during replay.
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        argmax_shard_publish_kernel<<<1, 256, 0, execution.dev[1]->stream>>>(
            workspace[1].partial, static_cast<ArgmaxCandidate*>(mailbox->payload(1, slot)),
            mailbox->flag(1, slot), workspace[1].tiles, columns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        argmax_shard_receive_kernel<<<1, 256, 0, execution.dev[0]->stream>>>(
            workspace[0].partial, workspace[0].local,
            static_cast<const ArgmaxCandidate*>(mailbox->payload(1, slot)),
            mailbox->flag(1, slot), mailbox->hang_word(), workspace[0].tiles, columns,
            static_cast<std::int32_t*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
        argmax_shard_finish_kernel<<<1, 256, 0, execution.dev[rank]->stream>>>(
            workspace[rank].partial, workspace[rank].local, workspace[rank].tiles, columns);
        CUDA_CHECK(cudaGetLastError());
    }
    // One-way, event-ordered pull. The peer's wait protects its source before the next
    // caller reuses that arena; rank zero consumes only its own copy after this point.
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(1), execution.dev[1]->stream));
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.dev[0]->stream, events.inputs_ready(1), 0));
    CUDA_CHECK(cudaMemcpyAsync(workspace[0].peer, workspace[1].local, bytes,
                               cudaMemcpyDeviceToDevice, execution.dev[0]->stream));
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), execution.dev[0]->stream));
    argmax_shard_merge_kernel<<<1, 256, 0, execution.dev[0]->stream>>>(
        workspace[0].local, workspace[0].peer, static_cast<std::int32_t*>(out.data), columns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
    CUDA_CHECK(cudaStreamWaitEvent(execution.dev[1]->stream, events.pull_done(0), 0));
}

} // namespace ninfer::ops::detail