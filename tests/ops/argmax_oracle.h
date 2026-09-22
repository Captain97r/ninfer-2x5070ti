#pragma once

#include "ops/op_tester.h"

namespace ninfer::test {

// Independent scalar scan over the complete logical column. Starting at row zero preserves
// its established NaN behavior; scanning increasing rows selects the first member of a tie.
inline std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                              std::int32_t physical_rows, std::int32_t tokens,
                                              std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best = 0;
        float best_value = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

} // namespace ninfer::test
