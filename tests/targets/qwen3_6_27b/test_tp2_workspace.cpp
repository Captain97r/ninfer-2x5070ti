// Arena-composition qualification, not a numerical kernel test. The byte oracle follows
// the real TP2 rank-zero allocation order, using public Op bounds at concrete shard shapes;
// it never calls the Variant leaf-capacity functions being tested or fits a measured peak.
#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "core/device.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using Variant    = targets::qwen3_6_27b::detail::Variant;
using Profile    = Variant::WeightsProfile;
namespace family = targets::qwen3_6::detail::qwen3_6_27b_runtime;

namespace {
void require(bool value, const char* what) {
    if (!value) throw std::runtime_error(what);
}

void bf(WorkspaceArena& a, int rows, int columns) { (void)a.alloc(DType::BF16, {rows, columns}); }

void i32(WorkspaceArena& a, int words) { (void)a.alloc(DType::I32, {words}); }

void scratch(WorkspaceArena& a, std::size_t bytes) {
    if (bytes == 0) return;
    auto scope = a.scope();
    (void)a.alloc_bytes(bytes);
}

std::size_t output_scratch(QType format, ops::LinearPolicy policy, int k, int t) {
    return std::max(ops::linear_add_workspace_capacity_bytes(format, 5120, k, policy, t, t),
                    ops::linear_workspace_capacity_bytes(format, 5120, k, policy, t, t));
}

void logits(WorkspaceArena& a, int t, bool optimized, bool modelopt) {
    auto scope = a.scope();
    bf(a, optimized ? 65536 : 124160, t);
    if (modelopt)
        scratch(a,
                ops::linear_workspace_capacity_bytes(QType::NVFP4_F32M, optimized ? 65536 : 124160,
                                                     5120, ops::LinearPolicy::CalibratedA4, t, t));
    scratch(a, optimized
                   ? ops::argmax_row_parallel_workspace_capacity_bytes(65536, t)
                   : std::max(ops::allgather_columns_workspace_capacity_bytes(124160, t),
                              ops::gather_columns_to_rank0_workspace_capacity_bytes(124160, t)));
}

void target(WorkspaceArena& a, int t, int batch, int width, bool prefill, bool record,
            ops::GqaExecutionEnvelope envelope, bool modelopt) {
    const auto fp8 = modelopt ? QType::FP8_E4M3FN_ROW_F32S : QType::FP8_E4M3FN_ROW_BF16S;
    const auto a8  = modelopt ? ops::LinearPolicy::CalibratedA8 : ops::LinearPolicy::AllowA8;
    {
        auto layer = a.scope();
        bf(a, 5120, t);
        bf(a, 3072, t);
        bf(a, 3072, t);
        bf(a, 512, t);
        bf(a, 512, t);
        scratch(a, ops::attn_input_proj_column_parallel_workspace_capacity_bytes(fp8, a8, t, t));
        bf(a, 3072, t);
        bf(a, 512, t);
        bf(a, 3072, t);
        scratch(a, ops::gqa_attention_workspace_capacity_bytes(12, DType::I8, envelope, batch,
                                                               width, width));
        scratch(a, output_scratch(fp8, a8, 3072, t));
    }
    {
        auto layer = a.scope();
        bf(a, 5120, t);
        (void)a.alloc(DType::FP32, {24, t});
        (void)a.alloc(DType::FP32, {24, t});
        bf(a, 3072, t);
        bf(a, 1024, t);
        bf(a, 1024, t);
        bf(a, 3072, t);
        scratch(a, ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(t, t));
        if (prefill) {
            bf(a, 5120, t);
            bf(a, 5120, t);
            scratch(a, ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(fp8, a8, t, t));
        } else {
            scratch(
                a, record
                       ? ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                             fp8, a8, batch, width, width)
                       : ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                             fp8, a8, batch, width, width));
        }
        // Both TP2 outputs exist before recurrent scratch, unlike the TP1 order.
        bf(a, 3072, t);
        bf(a, 3072, t);
        if (prefill) scratch(a, ops::gated_delta_net_workspace_capacity_bytes(8, 24, true, t, t));
        scratch(a, output_scratch(fp8, a8, 3072, t));
    }
    const std::vector<QType> mlp_formats =
        modelopt ? std::vector{QType::NVFP4_F32M} : std::vector{QType::NVFP4, fp8};
    for (const auto format : mlp_formats) {
        auto layer        = a.scope();
        const auto policy = modelopt                 ? ops::LinearPolicy::CalibratedA4
                            : format == QType::NVFP4 ? ops::LinearPolicy::AllowA4
                                                     : a8;
        bf(a, 5120, t);
        bf(a, 8704, t);
        scratch(a,
                ops::linear_swiglu_column_parallel_workspace_capacity_bytes(format, policy, t, t));
        scratch(a, output_scratch(format, policy, 8704, t));
    }
}

void mtp_stem(WorkspaceArena& a, int t) {
    // Rank zero allocates its embedding even when Vision supplies a composed replacement.
    for (int i = 0; i < 5; ++i) bf(a, 5120, t);
}

void mtp_post(WorkspaceArena& a, int t) {
    auto scope = a.scope();
    bf(a, 17408, t);
    bf(a, 8704, t);
    bf(a, 5120, t);
}

void mtp_core(WorkspaceArena& a, int t, int batch, int width, ops::GqaExecutionEnvelope envelope) {
    auto core = a.scope();
    bf(a, 5120, t);
    mtp_stem(a, t);
    bf(a, 3072, t);
    bf(a, 512, t);
    bf(a, 3072, t);
    bf(a, 512, t);
    {
        auto projection = a.scope();
        bf(a, 7168, t);
    }
    bf(a, 3072, t);
    bf(a, 512, t);
    bf(a, 3072, t);
    scratch(a, ops::gqa_attention_workspace_capacity_bytes(12, DType::I8, envelope, batch, width,
                                                           width));
    bf(a, 5120, t);
    bf(a, 5120, t);
    mtp_post(a, t);
}

void mtp_prefill(WorkspaceArena& a, int t, bool vision, bool optimized,
                 ops::GqaExecutionEnvelope envelope, bool modelopt) {
    auto call = a.scope();
    bf(a, 5120, t);
    bf(a, 5120, 1);
    bf(a, 5120, 1);
    bf(a, 5120, 1);
    {
        auto bulk = a.scope();
        mtp_stem(a, t);
        bf(a, 512, t);
        bf(a, 512, t);
        bf(a, 512, t);
    }
    bf(a, 3072, 1);
    bf(a, 3072, 1);
    bf(a, 3072, 1);
    bf(a, 5120, 1);
    bf(a, 5120, 1);
    bf(a, 3072, 1);
    if (vision) i32(a, 3);
    scratch(a, ops::gqa_attention_workspace_capacity_bytes(12, DType::I8, envelope, 1, 1, 1));
    mtp_post(a, 1);
    logits(a, 1, optimized, modelopt);
}

// Non-owning DeviceArena can account for a host-backed region: these tests enqueue no kernels
// and never pass these addresses to CUDA. Use a real aligned allocation, not invented pointers.
class ArenaCheck {
public:
    explicit ArenaCheck(std::size_t bound) : storage_(bound + 256), arena_(span(storage_, bound)) {}

    WorkspaceArena& arena() { return arena_; }

    void finish(std::size_t bound, const char* label) {
        require(arena_.used() == 0, "composition failed to restore its arena cursor");
        if (arena_.peak_used() > bound) {
            std::cerr << label << " peak=" << arena_.peak_used() << " bound=" << bound << '\n';
            throw std::runtime_error("composition exceeded planned phase capacity");
        }
        arena_.reset_peak();
    }
private:
    static DeviceSpan span(std::vector<unsigned char>& v, std::size_t bytes) {
        auto address = (reinterpret_cast<std::uintptr_t>(v.data()) + 255u) & ~std::uintptr_t(255u);
        return {reinterpret_cast<void*>(address), bytes};
    }

    std::vector<unsigned char> storage_;
    WorkspaceArena arena_;
};

void run(DeviceContext& device, int chunk, int batch_limit, int drafts, bool vision, bool optimized,
         std::uint32_t context, Profile profile = Profile::Qwen38Nvfp4) {
    const bool modelopt = profile == Profile::Qwen38ModelOpt;
    EngineOptions options;
    options.tp              = 2;
    options.devices         = {0, 1};
    options.max_context     = context;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(context);
    options.kv_cache        = KvCacheStorage::Int8Group64;
    options.prefill_chunk   = chunk;
    options.max_concurrency = batch_limit;
    options.enable_vision   = vision;
    if (drafts)
        options.speculative = {SpeculativeBackend::Mtp, static_cast<std::uint32_t>(drafts),
                               optimized ? ProposalHead::Optimized : ProposalHead::Full};
    auto planner       = targets::qwen3_6::make_sequence_planner<Variant>(device, options, profile);
    const auto pages   = planner.capacity_curve().minimum_main_page_groups;
    auto plan          = std::move(planner).finalize(pages);
    const auto& bounds = plan.impl_->workspace;
    ArenaCheck owner(bounds.capacity);
    auto& a = owner.arena();
    const ops::GqaExecutionEnvelope envelope{1, context};
    // Include scalar/small-T cutovers and odd final chunks, whose alignment differs from 1024.
    for (int t : {1,  2,  3,   4,   5,   6,   7,   8,   15,  16,  17,   31,
                  32, 33, 127, 128, 129, 255, 256, 511, 512, 513, 1023, 1024}) {
        if (t > chunk) continue;
        {
            auto root = a.scope();
            i32(a, t);
            i32(a, t);
            if (vision) i32(a, 3 * t);
            bf(a, 5120, t);
            if (vision) i32(a, t);
            bf(a, 5120, t);
            target(a, t, 1, t, true, false, envelope, modelopt);
            logits(a, 1, false, modelopt);
            scratch(
                a, ops::sampling_workspace_capacity_bytes(Variant::TextConfig::token_domain, 1, 1));
        }
        owner.finish(bounds.text_prefill, "text prefill");
        if (drafts) {
            auto root = a.scope();
            i32(a, t);
            i32(a, t);
            if (vision) i32(a, 3 * t);
            bf(a, 5120, t);
            if (vision) i32(a, t);
            bf(a, 5120, t);
            target(a, t, 1, t, true, false, envelope, modelopt);
            logits(a, 1, false, modelopt);
            scratch(
                a, ops::sampling_workspace_capacity_bytes(Variant::TextConfig::token_domain, 1, 1));
            i32(a, t);
            if (vision) {
                bf(a, 5120, t);
                i32(a, t);
            }
            mtp_prefill(a, t, vision, optimized, envelope, modelopt);
            if (drafts > 1) {
                auto iteration = a.scope();
                bf(a, 5120, 1);
                i32(a, 1);
                mtp_core(a, 1, 1, 1, envelope);
                logits(a, 1, optimized, modelopt);
            }
        }
        if (drafts) owner.finish(bounds.mtp_prefill, "MTP prefill");
    }
    for (int b = 1; b <= batch_limit; ++b) {
        {
            auto root = a.scope();
            bf(a, 5120, b);
            bf(a, 5120, b);
            target(a, b, b, 1, false, false, envelope, modelopt);
            logits(a, b, false, modelopt);
        }
        scratch(a, ops::sampling_workspace_capacity_bytes(Variant::TextConfig::token_domain, b, b));
        owner.finish(bounds.ordinary_round, "ordinary");
        if (!drafts) continue;
        const int width = drafts + 1, t = b * width;
        {
            auto root = a.scope();
            bf(a, 5120, t);
            bf(a, 5120, t);
            target(a, t, b, width, false, true, envelope, modelopt);
            logits(a, t, false, modelopt);
        }
        mtp_core(a, t, b, width, envelope);
        mtp_core(a, b, b, 1, envelope);
        logits(a, b, optimized, modelopt);
        scratch(a, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                       Variant::TextConfig::token_domain, drafts, drafts, b, b));
        scratch(a, ops::speculative_replicate_decision_workspace_capacity_bytes(drafts, b));
        owner.finish(bounds.mtp_round, "target verification / MTP / acceptance");
    }
    std::cout << "modelopt=" << modelopt << " chunk=" << chunk << " B=" << batch_limit
              << " K=" << drafts << " vision=" << vision << " optimized=" << optimized
              << " context=" << context << " text=" << bounds.text_prefill
              << " MTPprefill=" << bounds.mtp_prefill << " ordinary=" << bounds.ordinary_round
              << " MTPround=" << bounds.mtp_round << " vision_workspace=" << bounds.vision_encode
              << " capacity=" << bounds.capacity << '\n';
}
} // namespace

int main() {
    try {
        int count         = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0))
            return 77;
        CUDA_CHECK(status);
        DeviceContext device(0);
        if (device.sm() != 120) return 77;
        run(device, 1024, 1, 3, false, true, 196608);
        run(device, 1024, 1, 3, true, true, 196608);
        run(device, 128, 8, 5, false, false, 262144);
        run(device, 128, 8, 1, true, true, 131072);
        run(device, 1024, 1, 0, true, false, 196608);
        run(device, 1024, 1, 3, false, true, 196608, Profile::Qwen38ModelOpt);
        run(device, 1024, 1, 3, true, true, 196608, Profile::Qwen38ModelOpt);
        run(device, 128, 8, 5, false, false, 262144, Profile::Qwen38ModelOpt);
        run(device, 128, 8, 1, true, true, 131072, Profile::Qwen38ModelOpt);
        run(device, 1024, 1, 0, true, false, 196608, Profile::Qwen38ModelOpt);
        std::cout << "OK TP2 workspace composition\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
