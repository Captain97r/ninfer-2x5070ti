#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
    static constexpr int DecodePageSplitFloor = DecodeSplits;
    static constexpr int LongWindowSplits = DecodeSplits;
    static constexpr int LongWindowBegin = 0;
    static constexpr int LongWindowEnd = 0;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

// Head-local half of Gqa27Geometry for two-way tensor parallelism. Device r owns
// Q heads [r*12,(r+1)*12) and KV heads [r*2,(r+1)*2); its KV pool stores ONLY those two head
// pairs, so every KV head index the kernels compute is LOCAL (0 or 1) and every Q head index is
// local (0..11). GroupSize stays 6, so the Q-head -> KV-head grouping `qh / GroupSize` is the
// same arithmetic the tp1 geometry runs -- this is a geometry instantiation, not a new algorithm.
//
// DecodeSplitScale compensates the halved KV-head count in the split-KV decode grid, whose x
// extent is exactly KVHeads: 24|4 runs 4 x 85 = 340 CTAs, 16|2 runs 2 x 170 = 340, and 12|2 with
// scale 2 also runs 2 x 170 = 340. Each device drives its own GPU, so the per-device occupancy
// target is the whole 170-SM board, not half of one. This is the same inherited-tuning rule every
// split form here follows: the split reuses its family's measured policy rather than re-deriving
// one.
using Gqa27Tp2Geometry = GqaGeometry<12, 2, 2>;

// Measured INT8 T=1/4/5 profile for a single TP2 shard on a 70-SM sm_120 GPU.
// Keep the ordinary policy outside the qualified 81920..200709 visible-key interval.
// Broad replay envelopes still need the 170-split launch/workspace upper bound;
// the kernels select 70 active splits from the current device-side positions.
struct Gqa27Tp2Sm70Geometry : Gqa27Tp2Geometry {
    static constexpr int LongWindowSplits = 70;
    static constexpr int LongWindowBegin = 81920;
    static constexpr int LongWindowEnd = 200709; // 200704-token context plus five verify tokens
    // Conservative full-domain staging even when a replay chooses fewer active splits.
    static constexpr int DecodePageSplitFloor = LongWindowSplits;
};

static_assert(Gqa27Tp2Geometry::QHeads * 2 == Gqa27Geometry::QHeads);
static_assert(Gqa27Tp2Geometry::KVHeads * 2 == Gqa27Geometry::KVHeads);
static_assert(Gqa27Tp2Geometry::GroupSize == Gqa27Geometry::GroupSize);

// The complete registry. Launcher geometry selection is generated from this one list
// (ops/launcher/gqa_geometry_dispatch.cuh), so a newly registered geometry cannot be left behind
// in one launcher's hand-written switch. Q-head counts are pairwise distinct, which is what makes
// selection by Q-head count total and unambiguous.
#define NINFER_GQA_GEOMETRIES(X)                                                                   \
    X(Gqa27Geometry)                                                                               \
    X(Gqa35Geometry)                                                                               \
    X(Gqa27Tp2Geometry)

// The cache-append kernels (A2) read Geometry::KVHeads and nothing else -- no Q-head count, no
// group size, no split policy -- so any two geometries with the same KV-head count produce the
// same append kernel. Gqa27Tp2Geometry and Gqa35Geometry both carry 2, which makes selection by
// KV-head count over the full registry ambiguous: two entries would match. This sub-list resolves
// that by naming one representative per distinct KV-head count, and it is what the standalone
// `gqa_kv_append` entry point (which has no Q heads to select on) dispatches over, so that
// selection is a function and reordering either list cannot change which kernel runs.
//
// It is NOT a code-size mechanism: `gqa_attention_prompt_launch` reaches
// `gqa_kv_append_launch_for<Geometry>` through the full `dispatch_gqa_geometry`, so the append
// kernels are instantiated for every registered geometry regardless of what appears here.
// gqa_geometry_dispatch.cuh static_asserts that every entry of NINFER_GQA_GEOMETRIES has a
// representative here, so registering a geometry with a NEW KV-head count is a compile error until
// it is added.
#define NINFER_GQA_KV_REPRESENTATIVES(X)                                                           \
    X(Gqa27Geometry)                                                                               \
    X(Gqa35Geometry)

} // namespace ninfer::ops
