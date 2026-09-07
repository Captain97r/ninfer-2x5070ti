// Phase 3A-0/3A-1 measurement probe: prints the EXACT vision workspace / transient bytes
// the production planner would reserve, straight from the target's own layout code
// (targets/qwen3_6/impl/runtime/vision_context_impl.h). No CUDA device is required:
// every queried function is host-side arithmetic.
//
// Build (MSVC x64 + local CUDA 13.1 include paths; see scripts/build-vision-ws-probe.cmd):
//   cl /nologo /EHsc /O2 /std:c++20 /I src /I include /I third_party\cuda-13.1\include ^
//      tools\tp2\vision_ws_probe.cpp /link build-windows-131\src\ninfer_ops.lib
//      build-windows-131\src\ninfer_core.lib
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context_impl.h"

#include <cstdio>

int main() {
    using namespace ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime::schedule;

    struct Row {
        const char* label;
        std::uint64_t merged;
        std::uint64_t segments;
    };
    // (a) the production frozen reserve envelope: merged = min(capacity, 32768),
    //     segments = min(merged, 384) -- layouts_impl.h build_workspace_plan.
    // (b) realistic single items: one 48x48 image (2304 patches / 576 merged / 1 segment),
    //     a small image, a short video, a long video.
    const Row rows[] = {
        {"capacity 8k frozen envelope", 8192, 384},
        {"capacity 16k frozen envelope", 16384, 384},
        {"capacity 32k frozen envelope", 32768, 384},
        {"capacity 64k frozen envelope (capped 32768)", 32768, 384},
        {"capacity 100k frozen envelope (capped 32768)", 32768, 384},
        {"image 32x32", 256, 1},
        {"image 48x48 (max single image)", 576, 1},
        {"video 8x48x48", 2304, 8},
        {"video 16x48x48", 4608, 16},
        {"video 57x48x48 (merged 8192)", 8192, 57},
        {"video merged 2048", 2048, 16},
        {"video merged 4096", 4096, 32},
    };

    std::printf("%-42s %10s %9s %14s %12s %14s %12s\n", "case", "merged", "patches",
                "workspace B", "workspace", "output B", "output");
    for (const Row& row : rows) {
        const std::uint64_t patches = row.merged * 4;
        const std::size_t ws =
            VisionContext::workspace_capacity_bytes(static_cast<std::uint32_t>(row.merged),
                                                    static_cast<std::uint32_t>(row.segments));
        const std::size_t out = VisionContext::output_transient_bytes(row.merged);
        std::printf("%-42s %10llu %9llu %14zu %10.2f MiB %14zu %10.2f MiB\n", row.label,
                    static_cast<unsigned long long>(row.merged),
                    static_cast<unsigned long long>(patches), ws, ws / 1048576.0, out,
                    out / 1048576.0);
    }

    // Per-patch/per-token linear fit so the report can extrapolate.
    const std::size_t ws_small = VisionContext::workspace_capacity_bytes(1024, 8);
    const std::size_t ws_large = VisionContext::workspace_capacity_bytes(4096, 32);
    std::printf("\nper-merged-token workspace slope: %.1f KiB\n",
                (static_cast<double>(ws_large - ws_small)) / (3072.0 * 1024.0) * 1024.0);
    std::printf("output transient per merged token: 10240 B (5120 x BF16)\n");
    return 0;
}
