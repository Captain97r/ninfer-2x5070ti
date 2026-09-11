# Phase 3A-2 — Vision TP2 design (architecture E: dual-replicated, per-rank encode)

Basis: phase3a-vision-architecture.md (audit) + phase3a-vision-memory.md (feasibility).
Goal: `ninfer.exe MODEL.ninfer --tp 2 --devices 0,1 --vision ...` on Windows 11 WDDM,
no P2P, no NCCL, no text TP2 regression, mathematically verified against a single-GPU
oracle.

## 1. Why not tensor-parallel vision (per-operation analysis)

| Vision op | Shape (rows x cols) | TP2 strategy candidate | Collectives required | Verdict |
|---|---|---|---|---|
| patch_embedding | 1152x1536 + bias | column split (576 rows/rank) | allgather [1536,P] or partial-sum allreduce | linear is [1152,1536]: column-parallel splits K — each rank contracts half the 1536 patch features, needs partial allreduce [1152,P] per item |
| pos-embed add | [2304,1152] table | replicated | none | fine either way |
| attention qkv | 3456x1152 | head-split (8 heads/rank, 1728 rows) | none before attention | but vision_attention kernel grid.y is hardcoded 16 heads (vision_attention.cuh kVisionAttentionHeads) — new kernel + registered problem needed |
| attention out proj | 1152x1152 | row-parallel | allreduce [1152,P] after out-proj, per layer | 27 allreduces per item |
| mlp fc1 | 4304x1152 | column | none | fc1 2152/rank |
| mlp fc2 | 1152x4304 | row | allreduce [1152,P] per layer | 27 more |
| merger fc1/fc2 | 4608x4608 / 5120x4608 | column/row | allreduce [5120,V] | 1 more |
| **total** | | | **~55 allreduces/item** (sizes [1152,P] up to [5120,V]) | PeerMailbox slots are frozen 2048 sized for [5120,draft+1] decode payloads — vision payloads up to [5120,2048] exceed a slot and would fall back to the ~277 us staged path ~55x per item |

Layer split (C) analysis: the boundary activation is [1152, P] BF16 (5.3 MiB per 48x48
image) but the workspace does not halve (the per-item layout is a single live set, not a
per-layer sum), so C saves only 141 MiB of weights while requiring: a split VisionContext,
host-staged transfer of the boundary through pinned memory (no P2P), and strict
stream-ordering discipline across devices. Rejected.

Dual replication (E): every rank runs the bit-identical tp1 encode on its own device with
its own weights copy. ZERO collectives. The ONLY cross-device coupling is the existing
tp2 text machinery that consumes the embeddings afterwards.

## 2. Per-rank placement table (final)

| Tensor / buffer | Rank 0 (GPU0) | Rank 1 (GPU1) |
|---|---|---|
| vision weights (333 objects, 282.01 MiB) | full copy in device arena | full copy in device arena (NEW) |
| VisionContext (weights view + workspace) | existing | NEW instance |
| request transient (output [5120, merged]) | existing RequestMemory | NEW peer RequestMemory on execution.dev[1] |
| patch payload upload [P,1536] BF16 | per-item upload | per-item upload (same host source) |
| vision workspace envelope | capped 16,384-merged (tp2, raised from 2048 after VRAM re-measurement) | same |
| scatter into text x[r] | ops::scatter, rank 0 stream | ops::scatter, rank 1 stream |
| MTP stem visual overlap | scatter into rank 0 embedding root | none (rank 1 consumes hidden, not embeddings) |

Determinism: both ranks encode from the same host payload bytes with the same kernels and
same weights bytes — outputs are bit-identical by construction; verified in 3A-4.

## 3. Minimal change list (implementation order)

1. **Load/binder** (`qwen3_6_27b/impl/load/bindings.cpp`)
   - `bind_artifact`: drop the vision+tp2 throw; add `vision/` prefix (and
     `vision_merger` names) to the replicated family in `shard_mapping` so the tp2 binder
     places a full copy on every device.
   - `build_device_view`: drop the `tp != 1` throw; thread `device` through
     `materialize_vision_common` and merger fc2 materialization (`qwen3_6/impl/vision/bindings.cpp`).
2. **Guards**
   - engine.cpp `require_supported_tp_features`: remove the vision clause (keep DFlash).
   - layouts_impl.h `validate_target_options`: same removal.
   - CLI/serve help text + types.h comment updates.
3. **Workspace envelope policy** (layouts_impl.h build_workspace_plan)
   - tp2: `merged = min(capacity, kTp2VisionItemMergedLimit)` (16,384 since the 16 GB re-measurement); tp1 unchanged.
   - Existing request-plan check rejects oversized items loudly (unchanged).
4. **Request memory for rank 1**
   - `Qwen3_6_27BInstance` gains `request_memory_peer` bound to `execution.dev[1]`
     (registry.cpp); `concurrent_executor` activates both regions; `start_prefill_lane`
     receives both transients; `RequestControl::Prefill` gains `transient_peer`.
5. **VisionPrefillSession / VisionContext** (vision_context_impl.h)
   - Session holds an optional second VisionContext (peer device, peer weights view, peer
     workspace, peer transient). `prepare_chunk` encodes the item on BOTH ranks
     (stream-ordered per rank; timers synchronize per item) and returns
     `VisionChunk { embeddings[2] }`.
   - The per-item patches are uploaded from the same host payload to both devices.
6. **Text prefill tp2 multimodal** (text_context_impl.h)
   - `prefill_impl_tp2` accepts a `MultimodalPrefill`-style input (3-axis positions,
     vision session, begin, rope_delta):
     - rope_positions [3,len] uploaded per rank (same host source),
     - vision chunk embeddings per rank; scatter indices uploaded per rank;
       `ops::scatter` into x[r] on each rank,
     - MTP stem: visual-overlap scatter into rank 0's embedding root inside
       `mtp_forward_stem_tp2` (new optional argument), rank 1 unchanged.
7. **Prefix reuse**: vision-token suffix reuse at tp2 is downgraded to FullReset at plan
   time in phase 1 (documented limitation); tp1 reuse paths unchanged.

## 4. Correctness plan (3A-4)

- **Level 1 (component oracle)**: tools/reference VisionEncoder (torch, artifact-backed,
  standalone — loads only the 282 MiB vision objects) computes reference embeddings for
  the same processed patch payload on ONE GPU. The engine tp2 path dumps rank-0 and rank-1
  `VisionChunk.embeddings` for the same image; compare max/mean/relative error and expect
  bitwise equality between ranks (same kernels, same inputs) and near-BF16-tolerance vs
  the torch reference (same op order, different backend).
- **Level 2 (image+text E2E)**: greedy decode, fixed image+prompt at tp2 vs tp2 rerun
  (determinism) and vs the tp1 path where memory permits — the full 27B does NOT fit on
  one 16 GB GPU, so the tp1 comparison is component-level (Level 1) + determinism +
  qualitative grounding checks.
- **Level 3**: small/medium/large images, aspect ratios, multi-image prompt, image+video
  mix rejected gracefully (video over the 16,384-merged cap -> loud plan-time error).

## 5. Validation of the text path (3A-5/3A-6)

- Vision-off tp2 must remain byte-for-byte the same plan: `--tp 2` without `--vision`
  reserves identical memory (vision placement is opt-in through `features.vision`).
- Benchmarks before/after: MTP0/MTP3/MTP4 canonical runs (existing scripts), tolerance =
  measurement noise.
- Vision timings: preprocess, encode (both ranks), handoff (scatter), text prefill, TTFT.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Rank-1 vision weights silently absent (guard #3's original fear) | replicated placement is explicit in shard_mapping; `materialized_weight` re-checks placed bytes vs claimed shape at tp2 (existing mechanism) |
| Peer workspace exhaustion during concurrent prefill lanes | vision envelope is inside the planned per-rank workspace (same discipline as tp1 vision); encode is per-item synchronous before the text chunk reuses the arena |
| Peer request transient sizing | peer RequestMemory uses the same frozen capacity; activate() enforces the same bound |
| Vision regressions into decode | vision code is reachable only from prefill chunk prep; decode graph untouched; assert no vision ops appear in captured graph (graph node count unchanged) |
| Multi-lane concurrency + two transients | peer transient is per-Instance, activated per request exactly like rank 0's (same lifecycle, no new allocation paths) |
| Envelope cap rejects legitimate big videos at tp2 | loud, actionable error message names the cap; documented limitation; tp1 unaffected |
