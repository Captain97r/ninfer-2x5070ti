#pragma once

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/allreduce.h"
#include "ops/common/argmax_row_parallel_workspace.h"

namespace ninfer::ops::detail {

void argmax_row_parallel_launch(const std::array<Tensor, 2>& logits, Tensor& out,
                                std::int32_t valid_rows,
                                const std::array<ArgmaxRowParallelWorkspace, 2>& workspace,
                                const ExecutionContext& execution, const PeerTransfer& transfer);

} // namespace ninfer::ops::detail