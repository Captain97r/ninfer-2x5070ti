#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

// Multiplier-based artifacts preserve their exported FP32 scalars; legacy artifacts
// retain their original divisor arithmetic without a reciprocal conversion at import.
inline float nvfp4_product_multiplier(const Weight& weight) {
    return weight.qtype == QType::NVFP4_F32M
               ? weight.input_scale_multiplier * weight.weight_scale_multiplier
               : 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
}

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
