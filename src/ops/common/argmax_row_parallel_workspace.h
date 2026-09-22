#pragma once

#include "core/layout.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline constexpr int kArgmaxShardBlock = 512;

// One complete represented-value candidate per mailbox vector.
struct alignas(16) ArgmaxCandidate {
    float value;
    std::int32_t index;
    std::uint32_t reserved[2];
};
static_assert(sizeof(ArgmaxCandidate) == 16);

struct ArgmaxRowParallelWorkspace {
    ArgmaxCandidate* partial;
    ArgmaxCandidate* local;
    ArgmaxCandidate* peer;
    std::int32_t tiles;
};

struct ArgmaxRowParallelLayout {
    TensorRegion partial;
    TensorRegion local;
    TensorRegion peer;
    std::size_t bytes = 0;
    std::int32_t tiles = 0;

    [[nodiscard]] ArgmaxRowParallelWorkspace bind(DeviceSpan backing) const {
        return {static_cast<ArgmaxCandidate*>(partial.bind(backing).data),
                static_cast<ArgmaxCandidate*>(local.bind(backing).data),
                static_cast<ArgmaxCandidate*>(peer.bind(backing).data), tiles};
    }
};

inline ArgmaxRowParallelLayout make_argmax_row_parallel_layout(std::int32_t rows,
                                                               std::int32_t columns) {
    if (rows <= 0 || columns < 0) {
        throw std::invalid_argument("argmax_row_parallel: rows must be positive and columns nonnegative");
    }
    ArgmaxRowParallelLayout out;
    if (columns == 0) { return out; }
    out.tiles = 1 + (rows - 1) / kArgmaxShardBlock;
    LayoutBuilder layout;
    out.partial = layout.add_tensor(DType::I32, {4, out.tiles, columns}, 256,
                                     "argmax shard partials");
    out.local = layout.add_tensor(DType::I32, {4, columns}, 256, "argmax local candidates");
    out.peer = layout.add_tensor(DType::I32, {4, columns}, 256, "argmax peer candidates");
    out.bytes = layout.finish(256, "argmax row-parallel workspace");
    return out;
}

} // namespace ninfer::ops::detail