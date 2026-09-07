# Phase 3A-1 — Vision memory feasibility on 2 x RTX 5060 Ti 16 GB (Windows 11, WDDM)

All vision numbers are MEASURED (see phase3a-vision-architecture.md):
weights 282.01 MiB/full copy; workspace 67,744 B + output 10,240 B per merged token.
Text TP2 baseline is the measured load-summary from `logs/mtp0_run_err.txt`
(32k ctx, MTP0, vision off, `--kv-capacity auto`):

| per GPU | value |
|---|---:|
| text weights (nvfp4 shard) | 10.08 GiB |
| kv pool (auto) | 1.00 GiB |
| gdn state | 146.81 MiB |
| sequence pools | 1.16 GiB |
| workspace (text-only) | 182.81 MiB |
| cuda graphs | 2.00 MiB |
| **reserved total** | **11.43 GiB** |
| free / total (GPU0 / GPU1) | 2.96 / 3.29 GiB (15.93 GiB each) |

Usable per GPU ~15.93 GiB; GPU0 additionally carries the Windows desktop
(~0.6-0.9 GiB). KV at tp2: 128 KiB/token/GPU bf16 (64 layers x 2 KV heads x 256 x 2(K+V)
x 2 B), 64 KiB/token/GPU int8.

## 1. Architecture comparison at 32k context, bf16 KV (the binding case)

Vision costs per GPU under each architecture (weights W, workspace envelope E, output
transient O; `E32k` = uncapped 32k envelope = 2117 MiB + 320 MiB):

| Arch | GPU0 extra vs today | GPU1 extra vs today | Cross-GPU comm per item | Verdict |
|---|---:|---:|---|---|
| A. replicate on GPU0 (encode once, scatter both ranks via host copy) | +W +E32k +O = 2.66 GiB | +O (host-staged embeddings [5120,merged]) | merged embeddings D2H+H2D (~5.6 MiB per 48x48 image, ~2 ms) | Fits at 8k only; at 32k leaves 0.3 GiB — no margin; GPU0 is already the tight device |
| B. replicate on GPU1 | +O | +W +E32k +O | same as A but reversed | GPU1 has more headroom but GPU0 still pays the output + transfer staging; strictly worse than E for GPU0 |
| C. layer split (GPU0 layers 0..13, GPU1 14..26+merger) | +W/2 +E_l +O/2? see note | +W/2 +E_l +O | boundary activation [1152, P] BF16 once per item (~5.3 MiB per 48x48 image) | Workspace does NOT halve (each side runs one full live set; layout is per-item, not per-layer-sum); saves only W/2 = 141 MiB while adding a host-staged transfer + split VisionContext. Poor trade |
| D. tensor-parallel vision (heads 16->8, column QKV/fc1, row out/fc2/merger) | +W/2 +E/2 +O/2 ~ 1.33 GiB | same | ~55 allreduces per item (27 x 2 + merger fc2), each >= 1 mailbox round; new head-split kernel needed (grid.y hardcoded 16) | Highest complexity; changes numerics vs tp1 reference (reduction order); several ms added per item; kernel + dispatch work risks the validated ops registry |
| E. dual replication (each rank encodes the full item independently) | +W +E +O | +W +E +O | **zero** (no cross-GPU vision traffic at all) | See below |
| F. host offload (vision on CPU) | +O | +O | none (encode on CPU entirely) | Removes GPU cost but the Q4/Q5/W8 kernels are CUDA-only; a CPU path is a second implementation with its own correctness burden. Rejected |

With the UNCHANGED 32k envelope, E adds 2.66 GiB per GPU -> GPU0 ~14.1 GiB reserved
(0.3-0.6 GiB below the OOM line with the desktop). It "fits" only with zero practical
headroom — exactly what the task forbids ("do not accept a solution that works only
because it nearly exhausts VRAM").

**The envelope is the problem, not the replication.** The frozen envelope
`min(capacity, 32768)` merged tokens sizes the workspace for a single item that fills the
whole context with vision tokens. The planner already rejects any single item larger than
the envelope at request-plan time (request_plan_impl.h:146-148,
"vision item exceeds the Program workspace envelope"). The position table bounds a single
IMAGE at 48x48 = 576 merged tokens (37.21 MiB workspace). Only extreme videos approach the
uncapped envelope.

**Policy: at tp == 2 cap the frozen single-item envelope at 2048 merged tokens**
(8192 patches): workspace 132.31 MiB + output 20 MiB per GPU. Images always fit (max
576); videos up to ~7 frame-pairs at 48x48 (or many more at lower resolution, e.g. 24x24
-> 46 frame-pairs). Anything larger is rejected loudly at plan time by the existing check.
tp1 keeps `min(capacity, 32768)` unchanged — no upstream behavior change.

## 2. Feasibility table with the tp2 envelope cap (architecture E)

Per-GPU fixed cost with vision ON (weights + vision workspace delta over the 182.81 MiB
text-only workspace + output transient):

| Context | vision weights | vision workspace (capped) | output transient | total extra/GPU | GPU0 est. reserved | GPU0 est. free |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 282 MiB | max(183, 132) = 183 MiB (unchanged) | 80 MiB (cap: min(2048,cap)=2048 -> 20 MiB) | ~0.44-0.50 GiB | ~11.9 GiB | ~2.5 GiB |
| 32k | 282 MiB | 183 MiB (unchanged) | 20 MiB | ~0.50 GiB | ~11.9 GiB | ~2.5 GiB |
| 64k | 282 MiB | 183 MiB | 20 MiB | ~0.50 GiB | 11.9 + 4 GiB int8 KV = 15.9 | ~0 (bf16 impossible either way; int8 borderline) |
| 100k | 282 MiB | 183 MiB | 20 MiB | ~0.50 GiB | not feasible at any dtype (8 GiB int8 KV) | - |

(kv-pool auto at 32k reserved 1.0 GiB + sequence 1.16 GiB; a full-window 32k bf16 KV run
would add ~3 GiB — the same budget as text-only tp2, vision costs 0.5 GiB of it.)

Conclusion: with the envelope cap, vision at tp2 costs ~0.5 GiB per GPU vs vision-off and
the realistic context ceilings are IDENTICAL to text-only tp2 on this hardware:
- 8k-32k bf16: comfortable
- 64k: int8 KV only, borderline (same as text-only)
- 100k: not feasible (same as text-only)

The dominant VRAM term remains the 10.08 GiB text weights + KV, exactly as today.

## 3. Vision activation peak (per item, both ranks, E)

Peak live during encode of a 48x48 image (2304 patches): workspace allocations 37.21 MiB +
output 5.62 MiB + patch upload [2304,1536] BF16 6.75 MiB — ~50 MiB transient on top of the
frozen envelope, inside the planned arena. Two ranks encode the same item concurrently:
peak 2 x ~50 MiB active but on separate devices.

## 4. Decision

**Recommended architecture: E — dual-replicated Vision with per-rank independent encode,
plus a tp2-only single-item envelope cap of 2048 merged tokens.**

- Both GPUs hold the full 282.01 MiB vision tower and run the SAME item encode
  independently — bit-identical inputs, bit-identical kernels, bit-identical outputs,
  zero cross-GPU traffic. No new collectives, no PeerMailbox involvement, no new kernels,
  no numerics change vs the tp1 vision path.
- The text tp2 system is untouched except for the multimodal branch of prefill
  (scatter + 3-axis rope upload + MTP stem overlap), which is prefill-only and never
  enters the captured decode graph.
- Memory cost is dominated by weights (282 MiB) — 2.7% of the per-GPU text shard.
- The alternatives either do not reduce the binding device's footprint (A/B),
  save ~141 MiB while adding cross-GPU transfers and split contexts (C), or add ~55
  collectives per item, a new kernel, and reduction-order numerics drift (D).
