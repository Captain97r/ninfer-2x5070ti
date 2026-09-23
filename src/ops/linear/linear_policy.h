#pragma once

#include "ninfer/ops/linear.h"

#include <stdexcept>

namespace ninfer::ops::detail {

inline bool is_nvfp4_weight_type(QType qtype) {
    return qtype == QType::NVFP4 || qtype == QType::NVFP4_F32M;
}

inline bool is_fp8_weight_type(QType qtype) {
    return qtype == QType::FP8_E4M3FN_ROW_BF16S || qtype == QType::FP8_E4M3FN_ROW_F32S;
}

// Calibrated activation quantization belongs to the stored ModelOpt profile. It is
// mandatory for that profile and cannot be requested for legacy weight formats.
inline void validate_calibrated_linear_policy(QType qtype, LinearPolicy policy) {
    if ((qtype == QType::NVFP4_F32M) != (policy == LinearPolicy::CalibratedA4) ||
        (qtype == QType::FP8_E4M3FN_ROW_F32S) != (policy == LinearPolicy::CalibratedA8)) {
        throw std::invalid_argument("linear: calibrated policy does not match the weight profile");
    }
}

} // namespace ninfer::ops::detail
