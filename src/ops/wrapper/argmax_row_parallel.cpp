#include "ninfer/ops/argmax.h"
#include "ops/launcher/argmax_row_parallel.h"

#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

#ifndef NDEBUG
void require_device(const void* pointer, int device) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
            "argmax_row_parallel: storage is on the wrong device");
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes) {
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    require(a <= b ? first_bytes <= b - a : second_bytes <= a - b,
            "argmax_row_parallel: storage overlaps");
}
#endif

} // namespace

std::size_t argmax_row_parallel_workspace_capacity_bytes(std::int32_t local_physical_rows,
                                                         std::int32_t columns) {
    return detail::make_argmax_row_parallel_layout(local_physical_rows, columns).bytes;
}

void argmax_row_parallel(const std::array<Tensor, 2>& logits, Tensor& out,
                         std::int32_t valid_rows,
                         const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& execution, const PeerEvents& events) {
    require(execution.tp == 2 && execution.dev[0] && execution.dev[1] &&
                execution.dev[0]->device != execution.dev[1]->device,
            "argmax_row_parallel: two distinct devices are required");
    require(events.live(), "argmax_row_parallel: peer events must be live");
    const std::int32_t columns = logits[0].ne[1];
    for (const Tensor& shard : logits) {
        require(shard.dtype == DType::BF16 && shard.ne[0] > 0 && shard.ne[1] == columns &&
                    columns >= 0 && shard.ne[2] == 1 && shard.ne[3] == 1,
                "argmax_row_parallel: shards must be BF16 [rows,T] with matching T");
    }
    const std::int64_t rows = static_cast<std::int64_t>(logits[0].ne[0]) + logits[1].ne[0];
    require(rows <= std::numeric_limits<std::int32_t>::max() && valid_rows > 0 &&
                valid_rows <= rows,
            "argmax_row_parallel: invalid global row domain");
    require(out.dtype == DType::I32 && out.ne[0] == columns && out.ne[1] == 1 &&
                out.ne[2] == 1 && out.ne[3] == 1,
            "argmax_row_parallel: output must be I32 [T]");
    if (columns == 0) { return; }
    require(out.data && out.is_contiguous(), "argmax_row_parallel: invalid output storage");
    for (int rank = 0; rank < 2; ++rank) {
        require(logits[rank].data && logits[rank].is_contiguous() && workspace[rank],
                "argmax_row_parallel: invalid shard or workspace storage");
    }
    auto scope0 = workspace[0]->scope();
    auto scope1 = workspace[1]->scope();
    std::array<detail::ArgmaxRowParallelWorkspace, 2> scratch;
    for (int rank = 0; rank < 2; ++rank) {
        const auto layout = detail::make_argmax_row_parallel_layout(logits[rank].ne[0], columns);
        const DeviceSpan backing = workspace[rank]->alloc_bytes(layout.bytes, 256);
#ifndef NDEBUG
        require_device(logits[rank].data, execution.dev[rank]->device);
        require_device(backing.data, execution.dev[rank]->device);
        require_disjoint(logits[rank].data, logits[rank].bytes(), backing.data, backing.bytes);
        if (rank == 0) {
            require_device(out.data, execution.dev[0]->device);
            require_disjoint(out.data, out.bytes(), logits[0].data, logits[0].bytes());
            require_disjoint(out.data, out.bytes(), backing.data, backing.bytes);
        }
#endif
        scratch[rank] = layout.bind(backing);
    }
    detail::argmax_row_parallel_launch(logits, out, valid_rows, scratch, execution, events);
}

} // namespace ninfer::ops