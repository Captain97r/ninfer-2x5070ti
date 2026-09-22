#pragma once

#include "core/tensor.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

// Shared represented-input fixture and independent FP64 formula used by the descriptor-lifetime
// regression as well as the numerical profile tests.
std::vector<std::uint16_t> make_profile_activation(const Profile& profile, std::int32_t tokens);
std::vector<double> oracle_fp64(const Profile& profile,
                               const quantized_weight::PackedWeight& weight,
                               const std::vector<std::uint16_t>& activation, std::int32_t tokens);

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases);

} // namespace ninfer::test::linear_swiglu
