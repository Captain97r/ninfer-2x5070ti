# Phase 3A-0 — Vision architecture audit (qwen3_8_27b, Windows TP2 fork)

Scope: how Vision is wired today (tp 1), every place Vision + TP2 is rejected and why,
and the exact tensor/device facts the tp2 design must respect. All shapes/bytes below are
read from the actual artifact `M:\qwen\Qwen3.8-27b-nvfp4.ninfer` (333 vision objects,
295,711,648 bytes) and from the production layout code — not estimates.

## 1. Vision data path, end to end (tp 1 today)

```
HTTP/local media ─▶ media_acquire (FFmpeg decode)         [host, src/product/media_acquire]
                  ─▶ Frontend processor                   [host, src/targets/qwen3_6/impl/frontend/processor.*]
                       image → resize/crop/normalize → [raw_patches,1536] BF16 payload
                       grid = (t,h,w); patches = t*h*w; merged = patches/4
                  ─▶ chat template + tokenizer            [host; visual placeholder tokens in prompt]
                  ─▶ PreparedPromptData                   [token_ids, token_types, positions[3,T],
                                                          vision_items, media_payloads]
                  ─▶ RequestPlan (vision part)            [src/targets/qwen3_6/impl/runtime/request_plan_impl.h]
                       build_vision_control → VisionControl (per-item cu_seqlens,
                       position_ids, position table indices/weights, scatter_indices)
                       checks each item vs the Program workspace envelope (loud reject)
                  ─▶ VisionPrefillSession                 [runtime/vision_context_impl.h]
                       per item: upload patches → VisionContext::encode → [5120, merged] BF16
                       output transient; timers make encode synchronous
                  ─▶ TextContext::prefill (MultimodalPrefill)  [runtime/text_context_impl.h:1122]
                       per chunk: prepare_chunk(prompt_t0,len) → VisionChunk{embeddings, control}
                       scatter vision embeddings into x (token embedding) via ops::scatter
                       rope_positions uploaded as [3,len] (multimodal mrope)
                       MTP stem: visual-overlap shifted embeddings scattered into the MTP
                       input embedding (mtp_alignment.h windows; visual_scatter.h helper)
                  ─▶ text layers + decode                 [vision never runs again]
```

Key classes / responsibilities:

| Component | File | Responsibility |
|---|---|---|
| `VisionConfig` | `src/targets/qwen3_6_27b/impl/config.h` (via Variant) | dims: in 1536, hidden 1152, 27 layers, heads 16x72, mlp 4304, merge 2x2, merger 4608->5120 |
| `VisionBackboneConfig` | `src/targets/qwen3_6/export/.../qwen3_6/vision.h` | artifact-facing shape facts for binding |
| bind path | `src/targets/qwen3_6/impl/vision/bindings.cpp` | binds patch/pos embeddings, 27x(qkv,out,fc1,fc2,norms), merger fc1/fc2/norm |
| `VisionContext` | `src/targets/qwen3_6/impl/runtime/vision_context_impl.h` | one encode() per item: pos-embed add, 27 blocks (RMSNorm->QKV->flash attention->out proj->residual->RMSNorm->GELU MLP->residual), merger (norm->fc1->GELU->fc2) |
| `VisionPrefillSession` | same file | owns VisionContext + per-request staging; `prepare_chunk` returns per-chunk `VisionChunk` |
| `VisionControl` | `src/targets/qwen3_6/impl/vision/control.cpp` | per-item segment map, cu_seqlens, 2-D position ids + 4-point table interpolation, scatter indices |
| `ops::vision_attention` | `src/ops/launcher/vision_attention.cu` | packed non-causal 16-head x 72-dim flash, grid.y = heads (hardcoded 16), tiles for varlen segments |
| `ops::scatter`, `visual_scatter.h` | src/ops + runtime | token-embedding overwrite; MTP shifted-window scatter |
| `schedule::VisionContext::workspace_capacity_bytes` | runtime/vision_context_impl.h | frozen per-device workspace envelope builder |

## 2. Vision tensor map (actual artifact; full CSV: `research/phase3a-vision-memory.csv`)

333 objects, **295,711,648 bytes = 282.01 MiB** per full copy.

| Group | Objects | Bytes | Shapes |
|---|---:|---:|---|
| per-layer attention `qkv` | 27 | 57.11 MiB | Q4G64_F16S [3456,1152] |
| per-layer attention `output` | 27 | 23.50 MiB | Q5G64_F16S [1152,1152] |
| per-layer `mlp/fc1` | 27 | 71.18 MiB | Q4G64_F16S [4304,1152] |
| per-layer `mlp/fc2` | 27 | 81.36 MiB | Q5G64_F16S [1152,4304] |
| per-layer norms/biases | 27 x 8 | ~1.5 MiB | BF16 [1152]/[3456]/[4304] |
| `patch_embedding` (+bias) | 2 | 1.32 MiB | Q6G64_F16S [1152,1536], BF16 [1152] |
| `position_embedding` | 1 | 5.06 MiB | BF16 [2304,1152] (48x48 table) |
| merger `fc1` (+bias) | 2 | 21.51 MiB | W8G32_F16S [4608,4608] |
| merger `fc2` (+bias) | 2 | 23.90 MiB | W8G32_F16S [5120,4608] |
| merger norm | 2 | 4.5 KiB | BF16 [1152] |

Dtypes as stored: Q4G64_F16S 122.29 MiB, Q5G64_F16S 107.14 MiB, W8G32_F16S 45.42 MiB,
BF16 5.84 MiB, Q6G64_F16S 1.32 MiB. Runtime computes in BF16.

Measured workspace envelope (production `VisionContext::workspace_capacity_bytes`,
probe `tools/tp2/vision_ws_probe.cpp` -> `build/diagnostics/vision_ws_probe.exe`):

| Envelope (merged tokens) | Patches | Workspace (per GPU) | Output transient (per GPU) |
|---:|---:|---:|---:|
| capacity 8k | 32,768 | 529.25 MiB | 80.00 MiB |
| capacity 16k | 65,536 | 1058.50 MiB | 160.00 MiB |
| capacity >=32k (capped 32768) | 131,072 | 2117.00 MiB | 320.00 MiB |
| realistic image 48x48 (576) | 2,304 | 37.21 MiB | 5.62 MiB |
| realistic video 16x48x48 (4608) | 18,432 | 297.70 MiB | 45.00 MiB |

= 67,744 B workspace + 10,240 B output per merged token (66 KiB + 10 KiB).

Current device placement: **entire tower on device 0 only, replicated nowhere else**
(engine.cpp comment: "the Vision encoder runs entirely on the primary device"). Rank
assumptions: rank 0 owns VisionContext, workspace, request transient; rank 1 never sees a
vision tensor.

## 3. Every Vision+TP guard, and why it exists

| # | File / line | Guard | Why it exists | Correctness or memory? |
|---|---|---|---|---|
| 1 | `src/runtime/engine/engine.cpp:64-68` `require_supported_tp_features` | tp2 + enable_vision -> throw | Vision forward is single-device (weights land only on device 0 at tp2 because the binder has no replicated placement for `vision/*`), and the tp2 text prefill has no multimodal path (no 3-axis rope upload, no scatter, no MTP visual overlap) | correctness |
| 2 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:693-695` `validate_target_options` | same throw, target layer | target-level restatement of #1 | correctness |
| 3 | `src/targets/qwen3_6_27b/impl/load/bindings.cpp:843-848` `bind_artifact` | tp>1 + features.vision -> throw | the tp2 binder resolves EVERY object through `shard_placement`; `vision/*` names are not in the replicated list, so they would not be placed on device 1 (silent half-load) | correctness + memory |
| 4 | same file `:1074-1078` `build_device_view` | tp!=1 + vision -> throw | `materialize_vision_common`/`materialized_weight` take no `device` arg (default 0); rank 1's view would alias rank 0's arena | correctness |
| 5 | `src/targets/qwen3_6/impl/runtime/yarn_rope.cpp:112-117` | `--rope yarn` + `--vision` -> throw | vision ropes 2-D grid positions through its own 18-entry table; YaRN's 32-entry text table does not describe it — INDEPENDENT of tp, keep | correctness |
| 6 | `apps/cli/options.cpp:143-145` + `src/serve/serve_options.cpp:157,166` | help text "not --vision" | documentation only | - |
| 7 | `include/ninfer/types.h:96` | comment | documentation only | - |
| 8 | `apps/cli/options.cpp:303` + `serve_options.cpp:386` | `--spec dflash` + `--vision` -> throw | DFlash carry-own-rotary-table conflict — orthogonal, keep | correctness |

None of #1-#4 guards a *numerical* property of a split vision computation — they guard the
absence of (a) a load placement for rank 1 and (b) a multimodal tp2 prefill. Both are
addressable without touching PeerMailbox or text TP2 kernels.

## 4. Interaction with the existing TP2 text system

- `prefill_impl_tp2` (text_context_impl.h:1943) handles TextPrefill only: no
  `VisionChunk`, no 3-axis rope positions, no scatter, no MTP visual overlap.
- `mtp_forward_stem_tp2` (:2258) embeds on rank 0 (`ops::embedding`) and rmsnorms hidden on
  rank 1; a visual overlap must scatter into **rank 0's embedding root only** (rank 1 uses
  `normalized_hidden`, which already flows through the staged `hidden` input).
- Workspace plan (layouts_impl.h:598-604) adds `vision_encode` to the shared per-device
  workspace capacity — `PeerRuntime` allocates the same capacity, so rank 1's arena is
  already sized for a vision envelope the moment `features.vision` is true at tp2.
- Request transient (`RequestMemory`, bound to `execution.primary()`) is rank 0 only today;
  rank 1 has no request-scoped region. Vision output on rank 1 needs one.
- Decode path (CUDA Graph + PeerMailbox allreduces) never invokes vision — vision is
  prefill-only, so the validated decode system is untouched by design.
