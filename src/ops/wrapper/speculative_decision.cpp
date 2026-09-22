#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/speculative_decision.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_i32(const Tensor& tensor, std::int32_t rows, std::int32_t columns) {
    require(tensor.dtype == DType::I32 && tensor.ne[0] == rows && tensor.ne[1] == columns &&
                tensor.ne[2] == 1 && tensor.ne[3] == 1 && tensor.data && tensor.is_contiguous(),
            "speculative decision: field shape, dtype or storage is invalid");
}

#ifndef NDEBUG
void require_device(const void* pointer, int device) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
            "speculative decision: field or workspace is on the wrong device");
}

void require_disjoint(const void* first, std::size_t first_bytes,
                       const void* second, std::size_t second_bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    require(a <= b ? first_bytes <= b - a : second_bytes <= a - b,
            "speculative decision: state, config or scratch storage overlaps");
}
#endif

} // namespace

std::size_t speculative_replicate_decision_workspace_capacity_bytes(std::int32_t drafts,
                                                                    std::int32_t batch) {
    return detail::make_speculative_decision_layout(drafts, batch).capacity_bytes;
}

void speculative_replicate_decision(
    const std::array<SpeculativeDecisionView, 2>& decision, const SamplingConfig* peer_configs,
    const std::array<WorkspaceArena*, 2>& workspace,
    const ExecutionContext& execution, const PeerTransfer& transfer) {
    require(execution.tp == 2 && execution.dev[0] && execution.dev[1] &&
                execution.dev[0]->device != execution.dev[1]->device,
            "speculative decision: two distinct devices are required");
    require(transfer.matches(execution), "speculative decision: transfer belongs to another stream pair");
    require(peer_configs && workspace[0] && workspace[1],
            "speculative decision: peer config and both rank-local arenas are required");
    const int width = decision[0].licensed_tokens.ne[0];
    require(width >= 2 && width <= 6, "speculative decision: licensed width must be in [2,6]");
    const int batch = decision[0].licensed_tokens.ne[1];
    const auto layout = detail::make_speculative_decision_layout(width - 1, batch);
    for (int rank = 0; rank < 2; ++rank) {
        const auto& view = decision[rank];
        require_i32(view.frontiers, batch, 1);
        require_i32(view.anchors, batch, 1);
        require_i32(view.licensed_counts, batch, 1);
        require_i32(view.accepted_drafts, batch, 1);
        require_i32(view.licensed_tokens, width, batch);
#ifndef NDEBUG
        const std::array<const Tensor*, 5> fields{
            &view.frontiers, &view.anchors, &view.licensed_tokens,
            &view.licensed_counts, &view.accepted_drafts};
        for (std::size_t i = 0; i < fields.size(); ++i) {
            require_device(fields[i]->data, execution.dev[rank]->device);
            for (std::size_t j = i + 1; j < fields.size(); ++j) {
                require_disjoint(fields[i]->data, fields[i]->bytes(), fields[j]->data, fields[j]->bytes());
            }
            if (rank == 1) {
                require_disjoint(fields[i]->data, fields[i]->bytes(), peer_configs,
                                  sizeof(SamplingConfig) * batch);
            }
        }
#endif
    }
#ifndef NDEBUG
    require_device(peer_configs, execution.dev[1]->device);
#endif
    auto scope0 = workspace[0]->scope();
    auto scope1 = workspace[1]->scope();
    std::array<DeviceSpan, 2> scratch;
    for (int rank = 0; rank < 2; ++rank) {
        scratch[rank] = workspace[rank]->alloc_bytes(layout.capacity_bytes, 256);
#ifndef NDEBUG
        require_device(scratch[rank].data, execution.dev[rank]->device);
        const auto& view = decision[rank];
        for (const Tensor* field : {&view.frontiers, &view.anchors, &view.licensed_tokens,
                                    &view.licensed_counts, &view.accepted_drafts}) {
            require_disjoint(scratch[rank].data, scratch[rank].bytes, field->data, field->bytes());
        }
        if (rank == 1) {
            require_disjoint(scratch[rank].data, scratch[rank].bytes, peer_configs,
                              sizeof(SamplingConfig) * batch);
        }
#endif
    }
    detail::speculative_replicate_decision_launch(decision, peer_configs, scratch,
                                                  layout.transfer_bytes, execution, transfer);
}

} // namespace ninfer::ops
