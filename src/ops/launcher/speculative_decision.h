#pragma once

#include "ninfer/ops/speculative_round.h"
#include "ops/common/speculative_decision_workspace.h"

namespace ninfer::ops::detail {

void speculative_replicate_decision_launch(
    const std::array<SpeculativeDecisionView, 2>& decision, const SamplingConfig* peer_configs,
    const std::array<DeviceSpan, 2>& scratch, std::size_t transfer_bytes,
    const ExecutionContext& execution, const PeerTransfer& transfer);

} // namespace ninfer::ops::detail
