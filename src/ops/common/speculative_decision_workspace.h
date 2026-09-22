#pragma once

#include "core/layout.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct SpeculativeDecisionLayout {
    std::size_t transfer_bytes;
    std::size_t capacity_bytes;
};

inline SpeculativeDecisionLayout make_speculative_decision_layout(std::int32_t drafts,
                                                                  std::int32_t batch) {
    if (drafts < 1 || drafts > 5 || batch < 1 || batch > 8) {
        throw std::invalid_argument("speculative decision: requires K in [1,5] and B in [1,8]");
    }
    // Per row: every K+1 licensed slot, then frontier, anchor, licensed count and accepted count.
    // The transport includes initialized padding so every admitted shape fits mailbox granularity.
    const auto bytes = static_cast<std::size_t>(drafts + 5) * batch * sizeof(std::int32_t);
    const std::size_t transfer_bytes = (bytes + 15) & ~std::size_t{15};
    LayoutBuilder layout;
    (void)layout.add(transfer_bytes, 256, "speculative decision record");
    return {transfer_bytes, layout.finish(256, "speculative decision workspace")};
}

} // namespace ninfer::ops::detail
