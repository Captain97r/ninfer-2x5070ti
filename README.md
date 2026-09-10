# NInfer — Windows 11 TP2 fork

> Native Windows 11 tensor-parallel inference for Qwen3.8-27B NVFP4 on two consumer GeForce
> GPUs — measured on 2 × RTX 5060 Ti 16 GB, without requiring CUDA P2P, NCCL, WSL2 or Docker.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints.
It runs text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP
APIs. The 27B execution package can run tensor-parallel across two GPUs and, with YaRN positional
scaling, serves contexts up to 1,048,576 tokens -- see
[Dual-GPU (TP2) and YaRN 1M context](#dual-gpu-tp2-and-yarn-1m-context).

> **This is the Windows-TP2 experiment fork** ([ivanov84/ninfer-windows-tp2](https://github.com/ivanov84/ninfer-windows-tp2)),
> built on top of two other projects:
>
> - [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) -- the TP2/YaRN fork this
>   branch descends from (`6a355d5`), itself a fork of upstream [Neroued/ninfer](https://github.com/Neroued/ninfer)
>   (`feaf4dd`);
> - [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) -- the native Windows
>   (MSVC/vcpkg) port whose compatibility layer was cherry-picked here.
>
> What this branch adds on top of both:
>
> - a **native Windows 11 build of the TP2 engine** (no WSL2, no Docker, no NCCL);
> - a **pinned-host-memory peer-mailbox transport** for the small TP2 collectives, captured into
>   the decode CUDA graph, replacing the host-staged allreduce on systems where CUDA peer access
>   is unavailable -- see [Native Windows 11 TP2 on two GeForce GPUs](#native-windows-11-tp2-on-two-geforce-gpus)
>   and [docs/windows-peer-mailbox.md](docs/windows-peer-mailbox.md);
> - **vision (image input) working under `--tp 2`** -- a dual-replicated vision tower with
>   per-rank encode, a per-item pixel budget with automatic downscale, and MTP + image prefill
>   fixed -- see [Vision on TP2](#vision-on-tp2);
> - a second registered **Qwen3.8-27B NVFP4 artifact schema** (`nvfp4-split`) so the
>   [Ostfralla](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer) export loads on two
>   GPUs -- see [Tested Qwen3.8-27B NVFP4 NInfer artifacts](#tested-qwen38-27b-nvfp4-ninfer-artifacts).
>
> See [docs/PROVENANCE.md](docs/PROVENANCE.md) for the full layer-by-layer attribution and
> [NOTICE](NOTICE) for the required Apache-2.0 §4(b) attribution.

> **The base fork.** Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer); the TP2
> base of this tree branches from its commit `feaf4dd` and adds two things to the 27B execution
> package. **Dual-GPU tensor parallelism** (`--tp 2 --devices A,B`) halves per-card weight and KV
> residency and is ~40% faster at long context — one resident model, one process, two devices, no
> NVLink and no distributed serving. **YaRN ×4 positional scaling** (`--rope yarn`) raises the
> addressable ceiling from the registered 262,144 tokens to 1,048,576, computed to match vLLM as
> deployed and guarded by a drift test against the installed vLLM. Everything else is upstream's:
> `--tp 1` output is byte-identical to `feaf4dd` on the greedy cases in
> [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md), and single-GPU behaviour,
> supported identities, artifact format, and protocol surfaces are unchanged. The design
> decisions, numerical contracts, and qualification evidence behind both features are in
> [Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md).
> See [NOTICE](NOTICE) for attribution.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
| [Qwen3.8-27B NVFP4 — Ostfralla export](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer) | `nvfp4-split` | `qwen3_8_27b_nvfp4.ninfer` | 18,324,067,840 bytes (17.07 GiB) | `eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2` |

Qwen3.6-27B and Qwen3.8-27B each expose two registered weight profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four upstream 27B artifacts retain the same Text, Vision, MTP,
prefix-reuse, CLI, and serving routes.

The `nvfp4-split` row is the split-storage Qwen3.8-27B NVFP4 export published by Ostfralla: the same
checkpoint with separate GDN a/b control projections, BF16 early attention input projections and
W8G32 vocabulary planes. Its tensor schema differs from the neroued `nvfp4` artifact, so it is
accepted under its own registered identity; see
[Tested Qwen3.8-27B NVFP4 NInfer artifacts](#tested-qwen38-27b-nvfp4-ninfer-artifacts) for what was
verified on the two-GPU machine.

## Native Windows 11 TP2 on two GeForce GPUs

This fork is an experiment answering one question: **is tensor-parallel inference practical on
native Windows 11 with two consumer GeForce cards?**

On the tested system (2 × RTX 5060 Ti 16 GB, WDDM, asymmetric PCIe — see
[Hardware](#hardware-tested-windows-tp2-results)), `cudaDeviceCanAccessPeer(0, 1) == 0` — CUDA
peer access was not available between the two cards in this Windows 11 / WDDM configuration. That
is a measured property of the tested machine and driver, not a general statement about Windows.
The TP2 engine's fallback
transport was a host-staged allreduce: each of the ~128 collectives per decode round walked a
four-hop cross-device event chain plus driver-staged copy-engine transfers, at ~277 µs per 10 KiB
reduction — regardless of the fact that the useful payload only needs ~3.3 µs of PCIe time. The
communication *protocol*, not the PCIe bandwidth, was the bottleneck.

The fix in this fork is a **peer mailbox**: both devices exchange partial sums directly through a
pinned write-back host-memory slab via small CUDA kernels (publish → flag → poll → consume),
captured inside the existing cross-device CUDA graph. The staged path remains as the automatic
fallback for eager execution and oversized payloads. This is a transport for *small, frequent*
TP communication where WDDM/P2P limitations make conventional staged collectives expensive — it
is not a replacement for NCCL or a general claim of superiority over Linux.

### Key result (single request, Qwen3.8-27B NVFP4, greedy, thinking off)

| Configuration | Staged transport | Mailbox transport |
|---|---:|---:|
| MTP0 (no speculative decoding), 512 tok | 16.37 tok/s | **35.73–35.77 tok/s** (2.19×) |
| MTP3, 512 tok | 32.58 tok/s | **66.71–66.86 tok/s** (2.04×) |
| MTP4 (optimum), 512 tok | — | **68.53–68.87 tok/s** |
| MTP4, 1024 tok | — | 70.57–70.74 tok/s |
| MTP4, 2048 tok | — | 76.65 tok/s (acceptance 62.33%) |

Numbers are decode-time tok/s committed by the engine (not server/client wall-clock), measured as
described in [Benchmark methodology](#benchmark-methodology). The 2048-token figure is
**workload-dependent** — it rides a rising draft acceptance rate (53.06% → 62.33%) on repetitive
text and must not be quoted as a universal speed. All details, the per-MTP sweep, prompt-length
sensitivity, and the communication decomposition are in
[benchmarks/windows-tp2-benchmarks.md](benchmarks/windows-tp2-benchmarks.md) and
[docs/windows-peer-mailbox.md](docs/windows-peer-mailbox.md).

### Transport architecture: staged allreduce → pinned-host peer mailbox

Both transports compute the same thing — an exact BF16 elementwise sum of the two devices' partial
activations after each row-parallel projection. They differ in how many times the request crosses
the driver and the PCIe link.

**Old path (base-fork staged allreduce, one collective):**

```
GPU0 ─cudaMemcpyAsync→ host staging ─cross-device event chain (4 hops)→ host staging ─copy→ GPU1
GPU1 ─cudaMemcpyAsync→ host staging ─(same chain, second leg)───────────→ host staging ─copy→ GPU0
        each device then reads the peer partial and reduces
```

Four WDDM submission round trips and four driver-staged copies per reduction, ~128 reductions per
decode token. The useful payload needed ~3.3 µs of PCIe time; the choreography cost ~277 µs. Under
WDDM each cross-device submission pays a fixed driver cost independent of size, and the event
waits serialize the two devices' streams — the cards idle at 8–12% utilization waiting, not
transferring and not computing.

**Optimized path (this fork's peer mailbox, one collective):**

```
GPU0 ─store partial→ pinned WB host slot[0] ─flag→ poll peer flag ─read slot[1]─→ GPU0 reduces
GPU1 ─store partial→ pinned WB host slot[1] ─flag→ poll peer flag ─read slot[0]─→ GPU1 reduces
        both directions run concurrently inside the captured CUDA graph
```

One PCIe write, one flag round-trip, one PCIe read per device, all GPU-initiated from
`peer_exchange_sum_kernel` — no events, no copy engine, no driver round trip inside the exchange.
The mailbox operations are *captured into the decode CUDA graph* (slots are claimed once per call
site at capture time, so every replay fires every slot exactly once; the host resets flags between
replays and a bounded spin with an aggregate hang word prevents a silent deadlock inside WDDM's TDR
window). This removes the fixed per-collective synchronization/staging overhead rather than
accelerating the payload: a 40 KiB exchange costs the same transport floor (~17–19 µs) as a 10 KiB
one, of which only ~13 µs is payload time at Gen3 ×4.

Selection is conservative: the mailbox serves a collective only when `--tp 2` + CUDA graphs are
active, the stream is capturing, and the payload fits one slot; everything else — eager execution,
oversized payloads, `NINFER_TP2_MAILBOX=0` — takes the unchanged staged path. The full design,
correctness argument (exact-sum check against the staged path, bit-identical repeated runs), and
remaining bottlenecks are in [docs/windows-peer-mailbox.md](docs/windows-peer-mailbox.md).

### What was done

- **Windows port of the TP2 code** (from [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows),
  attributed in [docs/PROVENANCE.md](docs/PROVENANCE.md)): MSVC/vcpkg build,
  `CreateFileW`/`MapViewOfFile` artifact reader, winsock media acquire, MSVC-safe TMA
  descriptors and plan moves.
- **Peer mailbox transport** (new in this fork): `ops::PeerMailbox` — a pinned WB host slab
  (2048 slots, 256-byte aligned) + `peer_exchange_sum_kernel`, GPU-side publish/flag/poll/consume,
  installed at Program setup and claimed per captured call site; hang-guard and host-side flag
  reset between graph replays; `NINFER_TP2_MAILBOX=0` disables it.
- **Correctness**: exact-BF16-sum comparison against the staged path (`mailbox_probe`), eager and
  CUDA-graph modes, bit-identical repeated runs across 512/1024/2048-token generations, zero graph
  failures, zero fallback events in the final long run.
- **Vision on TP2** (new in this fork): the vision tower is bound twice, once per rank device, and
  each rank encodes the image locally into its own vision context, publishing the embeddings through
  the already-validated peer-mailbox publish path; the text TP2 prefill consumes them. The tp2
  vision workspace is sized for one item capped at 2048 merged tokens, and the frontend clamps
  `image_max_pixels`/`video_max_pixels` to that budget (2,097,152 px) so `smart_resize` downscales
  oversized media instead of the request plan rejecting it; MTP + image prefill gathers rope
  positions over the vision-merged axis; a request-scoped prefill failure now tears down only that
  request's lane instead of killing the worker. See [Vision on TP2](#vision-on-tp2).
- **Second Qwen3.8-27B NVFP4 artifact schema** (`nvfp4-split`, new in this fork): TP2 column-shard
  siblings of the BF16 fused attention input projection kernels plus the registered identity, so the
  split-storage Ostfralla export loads and serves on two GPUs. See
  [Tested Qwen3.8-27B NVFP4 NInfer artifacts](#tested-qwen38-27b-nvfp4-ninfer-artifacts).

### Vision on TP2

Image input works under `--tp 2` in this fork — the base fork rejected `--tp 2 --vision` at startup
because its vision encoder had no split path. What "vision on TP2" means here:

- **Dual-replicated tower, per-rank encode** (architecture E of the phase-3A audit): each rank
  materializes its own full copy of the vision weights (282 MiB per GPU) and encodes the image
  locally with the unmodified tp1 vision code, so there is zero cross-GPU traffic during encode.
  The resulting embeddings are published to the peer through the existing validated
  `PeerMailbox` publish path and consumed by the text TP2 prefill — no CUDA P2P, no NCCL.
- **Vision is prefill-only**: encode runs once per image prompt (measured 0.047–0.048 s per image)
  and never enters the MTP decode loop or the captured decode graph; decode speed after an image
  prompt is identical to the text-only TP2 path.
- **Per-item pixel budget with automatic downscale**: at `--tp 2` the vision workspace is sized for
  one item capped at 2048 merged tokens (2,097,152 px after `smart_resize`; one merged token = a
  32 × 32 pixel block). The frontend clamps the preprocessor's `image_max_pixels` and
  `video_max_pixels` to exactly that budget, so a 15.75 MP photo is downscaled before encoding
  instead of being rejected by the plan-time envelope check; `--tp 1` keeps upstream behavior.
- **MTP + image prompts work**: the tp2 draft stage gathers rope positions over the vision-merged
  axis (mirroring the tp1 final-chunk stage), so `--spec mtp` with any image no longer crashes
  mid-prefill with non-contiguous positions. Measured MTP acceptance on image prompts (server
  smoke test, three image requests): 2.80–3.32 tokens/round (60–80%).
- **A failed request no longer kills the worker**: a request-scoped prefill exception tears down
  only that request's lane (abort/reset/complete-error) and serving continues; a sticky CUDA device
  error still escalates to a full engine stop.

Measured on 2 × RTX 5060 Ti 16 GB, Qwen3.8-27B NVFP4 (`nvfp4` artifact), single request, greedy,
CUDA graphs on (`research/phase3a-vision-validation.md`):

| Vision workload | Measured |
|---|---:|
| vision encode, per image (chart / natural photo) | 0.047–0.048 s |
| text prefill including image embeddings | 806–824 tok/s over 457–468 prompt tokens |
| time to first token, 468-token multimodal prompt | ~0.62 s |
| decode after an image prompt (MTP0) | 35.65–35.73 tok/s (unchanged from text-only) |
| text TP2 regression on the same tree | MTP0 35.83 / MTP3 66.68 / MTP4 68.86–68.87 tok/s |
| per-GPU memory cost of vision | 282 MiB weights + ~0.5 GiB total incl. workspace/transient |

Determinism: reruns with identical parameters produced bit-identical output (SHA-256 compared
across runs). The published tp2 validation covers still-image prompts through both the CLI
(`scripts/phase3a-vision-verify.cmd`) and the server (image requests with `--spec mtp`, small and
auto-downscaled 15.75 MP images, with the engine serving follow-up text requests afterwards).
Video passes through the same clamped preprocessor path but was not part of the published tp2
validation. Vision remains mutually exclusive with `--rope yarn` (the encoder ropes 2-D image-grid
positions through its own table), and `--spec dflash` remains text-only.

Run an image prompt through the CLI on two GPUs:

```bat
build-windows\apps\ninfer.exe models\qwen3_8_27b_nvfp4.ninfer --tp 2 --devices 0,1 ^
  --vision --greedy --max-new 128 ^
  --messages examples\cli\messages\image_chart.json
```

The committed design and audit behind this path are in
[research/phase3a-vision-architecture.md](research/phase3a-vision-architecture.md),
[research/phase3a-vision-memory.md](research/phase3a-vision-memory.md),
[research/phase3a-vision-tp2-design.md](research/phase3a-vision-tp2-design.md) and
[research/phase3a-vision-validation.md](research/phase3a-vision-validation.md).

### Tested Qwen3.8-27B NVFP4 NInfer artifacts

Two published Qwen3.8-27B NVFP4 NInfer artifacts have been tested with this Windows TP2
implementation on the 2 × RTX 5060 Ti 16 GB machine. No claim is made about other Qwen3.8
artifacts; the engine accepts only the registered identities in the table above.

| Artifact | Identity | Size | Tested with |
|---|---|---:|---|
| [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3.8-27b` / `nvfp4` | 21,492,695,040 bytes (20.02 GiB) | the primary artifact: every Windows-TP2 benchmark on this page, plus the full vision validation above |
| [Ostfralla/Qwen3.8-27B-NVFP4-NInfer](https://huggingface.co/Ostfralla/Qwen3.8-27B-NVFP4-NInfer) | `qwen3.8-27b` / `nvfp4-split` | 18,324,067,840 bytes (17.07 GiB) | functional TP2 serving smoke test: loads both GPUs and serves at `--max-context 131072` with MTP3 (`--lm-head-draft`) |

The Ostfralla export stores the same checkpoint with a different tensor layout (separate GDN a/b
control projections, BF16 early attention input projections, W8G32 vocabulary planes). Because the
engine selects the binding schema from the manifest identity — and the plain `nvfp4` identity is
already taken by the neroued mixed FP8/NVFP4 layout — this fork registers that layout as
`nvfp4-split` and adds the TP2 column-shard kernels it needs (the BF16 fused attention input
projection at the halved shard geometry). The published Ostfralla file declares `nvfp4` in its
manifest; loading it with this engine requires that identity field to read `nvfp4-split`. The
tested copy differs from the published file only in those six manifest bytes — the weight payload
is byte-identical (verified by comparing the manifest object tables and sampled payload ranges at
identical offsets across the whole file, plus the file tail).

### Limitations

- Tested on **one machine** (see [Hardware](#hardware-tested-windows-tp2-results)); no claim of
  portability to other GPUs, drivers, or PCIe topologies.
- Consumer GeForce / WDDM specific behaviour: no CUDA P2P on the tested configuration, and the
  mailbox is what makes the fallback path cheap here. Linux/NCCL systems may behave completely
  differently; **no Linux comparison is claimed**.
- The mailbox installs whenever `--tp 2` and CUDA graphs are enabled, regardless of whether P2P
  works on the host — on P2P-capable systems (e.g. the 2 × RTX 5090 the base fork targets) the
  direct path may be preferable and `NINFER_TP2_MAILBOX=0` restores the staged transport.
- MTP performance depends on draft acceptance rate; long-generation tok/s differs from
  512-token benchmarks.
- Vision at `--tp 2` is sized for one media item per request capped at 2048 merged tokens
  (2,097,152 px); larger media is downscaled to that budget, and an item that still exceeds the
  envelope after preprocessing is rejected at request-plan time with a clear error. YaRN remains
  mutually exclusive with `--vision` at any `--tp`.
- Remaining bottleneck after the mailbox is synchronization/lockstep waiting plus memory-bound
  GEMM (~60% of a round), not useful PCIe bandwidth — the 40 KiB exchanges sit at the ~17–19 µs
  transport floor while only ~13 µs is payload time.

## Performance

The published measurements cover the three Qwen3.6 artifact profiles and the Qwen3.8-27B NVFP4
profile. The Qwen3.8-27B `groupwise-int` profile is supported by current NInfer builds but is not
yet included in a published benchmark campaign.

### Concurrent MTP3 decode

Saturated decode was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, MTP3, and
one 8,192-token generation per active request. The values below are aggregate committed decode
throughput from complete one-second intervals in which the actual decode batch remained equal to
the configured concurrency. MTP acceptance is aggregated over the complete request wave. Each
concurrency cell reports `decode tok/s / MTP acceptance`; profiles should be read independently.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

At C=8, Qwen3.6-35B-A3B reaches **1,313.8 aggregate decode tok/s**. Qwen3.6-27B NVFP4 reaches
**1,146.9 tok/s** and **5.67×** its C=1 throughput. Qwen3.8-27B NVFP4 has **45.8–48.9%** MTP
acceptance, versus **67.2–71.4%** across the other measured profiles, so aggregate committed
throughput reflects both execution performance and speculative acceptance.

### Single-request serving

The single-request corpus was measured on the same GPU with INT8 group-64 KV cache, CUDA Graphs,
and a 1,024-token prefill chunk. Each reported fixture uses five fixed seeds after server warm-up.
Targets and weight profiles are reported independently rather than as cross-target comparisons.
Requests were submitted serially to a persistent server. The Qwen3.8-27B NVFP4 MTP0 results use the
same dedicated serial corpus runner as the Qwen3.6 profiles; its MTP3 results come from the C=1 point
of the fixed concurrent-corpus campaign documented in [Performance](docs/performance.md).

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **620.3–726.2 decode tok/s** with **72.7–82.8% acceptance**.
- MTP3 structured output: **770.9 decode tok/s**, **89.1% acceptance**, and **3.67 tokens/round**.

**Qwen3.6-27B (`groupwise-int`)**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **161.9–175.4 decode tok/s** with **73.4–78.8% acceptance**.
- MTP3 structured output: **193.0 decode tok/s**, **88.7% acceptance**, and **3.66 tokens/round**.

**Qwen3.6-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **11,191.5 prefill tok/s** and **86.4 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,510.6 prefill tok/s** and **59.9 decode tok/s**.
- MTP3 long reasoning: **213.1–231.0 decode tok/s** with **76.3–81.1% acceptance**.
- MTP3 structured output: **252.2 decode tok/s**, **89.8% acceptance**, and **3.69 tokens/round**.
- Against groupwise-int on the same corpus and runtime options: **3.48× the 7,680-token prefill
  throughput**, **1.55× the 260,096-token prefill throughput**, and **30–32% higher MTP3 decode
  throughput**.

**Qwen3.8-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **8,340.4 prefill tok/s** and **71.2 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,203.1 prefill tok/s** and **52.9 decode tok/s**.
- MTP3 long reasoning: **151.4–195.2 decode tok/s** with **56.2–76.0% acceptance**.
- MTP3 structured output: **219.8 decode tok/s**, **90.8% acceptance**, and **3.72 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8-27B rows used
temperature 1.0 and presence penalty 0.0. The multimodal columns (ERQA and RealWorldQA) ran with
`--vision` at a 81,920-token context limit; the text columns used a 262,144-token limit except
Qwen3.8-27B NVFP4, which needs 252,928 to fit the RTX 5090 after weights.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- one NVIDIA GeForce RTX 5090 (`sm_120a`), or two for `--tp 2`;
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. There is no install target or packaged
binary distribution; NInfer is run from its source build tree.

## Build

Clone this fork, not upstream — upstream has neither `--tp 2` nor `--rope yarn`.

```bash
git clone https://github.com/wamansou/ninfer-tp2-1m.git
cd ninfer-tp2-1m

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Docker

Build the runtime image on a 64-bit Linux host with an RTX 5090, a CUDA 13.1-compatible NVIDIA
driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build --tag ninfer:local .
```

Download a model into `models/` as described below, then run the HTTP server:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_6_27b.ninfer \
  --host 0.0.0.0
```

Run the CLI from the same image:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer /models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 256
```

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models

# Or the split-storage Qwen3.8-27B NVFP4 export (Windows-TP2 fork; see
# "Tested Qwen3.8-27B NVFP4 NInfer artifacts" for the identity relabel it needs):
hf download Ostfralla/Qwen3.8-27B-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all six downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
available only for the 35B-A3B target and is text-only.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Dual-GPU (TP2) and YaRN 1M context

`--tp 2` splits one resident model across two RTX 5090s, and `--rope yarn` raises the addressable
context ceiling from the registered native 262,144 tokens to 1,048,576. The two features are
independent -- TP2 halves per-card weight and KV residency at any context, YaRN extends positions
at either `--tp` width -- but 1,048,576 tokens only fits when both are used together with INT8 KV.

TP2 is a capacity feature, not a scale-out feature: one process, one resident model, two CUDA
devices, no NVLink and no distributed serving. It is implemented for the 27B execution package
(`qwen3.6-27b` and `qwen3.8-27b`, either weight profile); `qwen3.6-35b-a3b` has no tensor-parallel
path and rejects `--tp 2` at startup. Every measurement below was taken on the Qwen3.8-27B NVFP4
artifact.

### Usage

```bash
# 1,048,576-token context, both GPUs, INT8 KV, MTP3 speculative decoding
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The same flags drive the CLI; add `--no-thinking` when a short `--max-new` budget must reach the
answer channel, because at this checkpoint's default thinking mode a 16-token budget is consumed
entirely inside the reasoning stream:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

- `--tp 2` requires an explicit `--devices A,B` naming two distinct devices of the same compute
  capability. `--tp 1` remains the default and is unchanged: `scripts/tp1-regression.sh` compares
  this build's greedy output against token streams recorded from upstream `ninfer` at base commit
  `feaf4dd`, over a short chat, a 2,191-token instruction and a 28,677-token document, and requires
  the token ids, the generated text and the deterministic summary rows to be byte-equal. Its scope
  is exactly that: greedy text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP
  off, concurrency 1. See `tests/data/tp1-golden/MANIFEST.md`.
- `--rope yarn` needs `--yarn-origin` to equal the artifact's registered native capacity
  (`262144`); `--yarn-factor` is a finite value in `[1.0, 64.0]` and `origin x factor` must be a
  whole token count at or below `1048576`.
- `--kv-dtype int8` is mandatory at 1M: BF16 KV needs four times the pool and does not fit.
- `--max-concurrency 1` is arithmetic, not policy, at 1M -- one sequence costs 16.66 GiB per device
  without MTP and 17.69 GiB with it, so a second slot cannot fit on a 32 GiB card.
- MTP speculative decoding (`--spec mtp --draft-tokens 1..5`, optionally `--lm-head-draft`) works
  at `--tp 2` including at 1M. In this fork `--vision` also works at `--tp 2` (see
  [Vision on TP2](#vision-on-tp2)); `--spec dflash` is rejected at `--tp 2`.

See the [CLI guide](docs/cli.md) and [HTTP serving](docs/serving.md) for the full option contract.

### Memory, per GPU

Qwen3.8-27B NVFP4, `--tp 2 --devices 0,1 --kv-dtype int8 --kv-capacity auto`, one active request.
`Resident` is `nvidia-smi` per-process memory, which was flat across every sample of every run --
a 1M-token prefill does not push residency above the value chosen at load.

| Context | MTP | Weights | Sequence (KV + GDN state) | Workspace | Reserved (load summary) | Resident (`nvidia-smi`) |
|---:|---|---:|---:|---:|---:|---:|
| 262,144 | off | 10.08 GiB | 4.28 GiB | 0.18 GiB | — | **15.04 GiB** |
| 262,144 | MTP3 | 10.46 GiB | 4.54 GiB | 0.19 GiB | — | **15.69 GiB** |
| 1,048,576 | off | 10.08 GiB | 16.66 GiB | 182.81 MiB | 26.93 GiB | **27.41 GiB** |
| 1,048,576 | MTP3 | 10.46 GiB | 17.69 GiB | 192.93 MiB | 28.42 GiB | **28.84 GiB** |

The KV pool is 16.5 KiB per token per device at INT8 group-64, quantization-scale planes included,
which is the 16.50 GiB pool at 1,048,576 tokens.
Turning MTP3 on costs a **measured 1.49 GiB of reserved memory per device at 1M and 0.65 GiB at
262k** -- it is not a constant. Its two dominant terms are **0.38 GiB of head weights**, fixed at
any window, and **1.03 GiB of MTP KV per 1M tokens of window**; the remainder of each measured
delta is workspace and sequence-arena rounding. The KV term is one full extra attention layer's
worth of KV over the window -- one sixteenth of the text pool -- which is what a one-layer MTP head
costs. The 262,144-token rows were measured through `ninfer-serve`'s startup record and
`nvidia-smi`, which is why they carry no CLI load-summary `reserved` row.

At 1M with MTP3 the headroom to a 30 GiB per-device budget is **1.16 GiB**, the tightest shipped
configuration. For comparison, the same artifact at `--tp 1` needs 27.9 GiB on one card for a
252,928-token window with MTP off and 29.16 GiB with MTP3, and cannot reach 262,144 at all.

### Measured performance

**Every figure carries a per-GPU power limit.** The campaign was measured at a **400 W** cap (the
minimum settable limit on these cards); the publishable set was then re-measured with both cards at
**575 W** (vendor defaults are 600 W and 575 W; maximum 600 W). Do not quote any figure below
without its power condition.

Single request, INT8 KV, CUDA Graphs on, greedy decoding, byte-identical prompt on both widths:

| Workload | TP1 @400 W | TP2 @400 W | TP1 @575 W | TP2 @575 W |
|---|---:|---:|---:|---:|
| 249,955-token prompt, MTP off — prefill | 2,269.8 | **2,680.1** | 2,484.2 | **2,787.0 tok/s** |
| 249,955-token prompt, MTP off — decode | 52.35 | **75.18** | 53.95 | **75.32 tok/s** (1.40x) |
| 249,955-token prompt, MTP3 — decode | 101.7 | **152.1** | 113.60 | **159.39 tok/s** |
| 249,955-token prompt, MTP3 — draft acceptance | 50.83% | **57.96%** | 50.83% | **57.96%** |
| 249,955-token prompt — time to first token | 111.0 s | **93.7 s** | 101.0 s | **90.0 s** |
| 536-token reasoning prompt, MTP3 — decode | 152.0 | **189.5** | — | — |
| 652,955-token prompt, MTP off — prefill / decode | does not fit | **1,348.4 / 58.22** | does not fit | not re-measured |
| 1,045,954-token prompt, MTP off — prefill / decode | does not fit | **928.9 / 46.39** | does not fit | **975.1 / 48.08** |
| 1,045,954-token prompt, MTP3 — decode (512 tokens) | does not fit | — | does not fit | **100.54 tok/s** at 56.41% |

Lifting the cap helps TP1 more than TP2, and the reason is visible in sampled draw: one card running
the whole model saturates its limit (peak 575.5 W), while two cards sharing it peak at 391 and
406 W. The TP2-over-TP1 decode advantage therefore narrows from **1.44x at 400 W to 1.40x at
575 W** -- TP2 is still faster, and it gets there inside a much smaller power envelope.

At 575 W the 1,045,954-token prefill takes **17.9 minutes** per request, against 18.8 at the 400 W
cap. Decode degrades smoothly with context rather than falling off a cliff: at the 400 W cap,
97.9 tok/s at 8k, 58.2 at 653k and 46.4 at 1,046k; the 1,046k point rises to 48.1 at 575 W. Only
the 250k and 1,046k tiers were re-measured at 575 W. At long context TP2 is *faster* than TP1, not
merely larger, because per-card weight and KV traffic halve while the cross-device collectives cost
about 0.2 ms per token under CUDA Graphs.

Saturated concurrent decode at a 262,144-token window, `--tp 2`, aggregate committed tok/s:

| Concurrency | MTP off @400 W | MTP3 @400 W | MTP off @575 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|
| 1 | 93.5 | 172.4 | 94.1 | 177.8 |
| 2 | 183.0 | 287.0 | — | — |
| 4 | **314.3** (3.36x) | **466.0** (2.70x) | **320.3** (3.40x) | **475.2** (2.67x) |

Per-GPU residency stayed at or below 15.64 GiB at C=4.

### Compared with vLLM

On byte-identical 250k / 653k / 700k-token prompts with **both cards capped at 500 W** -- a third
power condition, not comparable with the 400 W and 575 W rows above -- vLLM 0.25.1 (TP2, FP8 KV,
YaRN through `--hf-overrides`, MTP3) prefills **1.17-1.32x faster**, while NInfer decodes **1.41x
faster at 250k and 2.5-2.8x faster at 653k and 700k**, because vLLM's MTP acceptance is exactly
**0%** past its native 262,144-token window where NInfer's stays at 51-60%. NInfer also holds a
**38% larger window in less memory**: 1,048,576 tokens at 27.41 GiB per GPU against vLLM's 759,297
at 28.90 GiB. At a 512-token answer the request is almost all prefill, so vLLM finishes first at
every tier; NInfer wins beyond roughly 4,800 / 7,600 / 8,800 output tokens. The two engines ran
different NVFP4 quantizations of different fine-tunes and different KV dtypes, and no quality claim
is made -- the full table, method and caveats are in
[Performance](docs/performance.md#cross-engine-comparison-against-vllm-nvfp4-500-w-per-gpu).

### Retrieval

Needle-in-a-haystack, five depths (10/30/50/70/90%) x two independent haystacks, rule-scored exact
match, thinking disabled, greedy:

| Engine and configuration | Haystack | Prompt tokens | Retrieved |
|---|---|---:|:-:|
| NInfer TP2, native rope | 262k, **tiled** corpus | 259,954 | **10 / 10** |
| NInfer TP2 + YaRN x4 | 653k, distinct text | 652,954-652,955 | **10 / 10** |
| NInfer TP2 + YaRN x4 | **1,046k, distinct text** | 1,045,954-1,045,955 | **10 / 10** |
| vLLM control (model ceiling) | 653k, distinct text | 652,954-652,955 | **10 / 10** |

Three qualifications travel with that table:

- **The 262k row uses a tiled haystack.** The stock corpus is 2.7 MB (644 KB of English essays plus
  a Chinese novel), so any English tier past roughly 150k tokens repeats the corpus -- at 262k each
  window occurs about twice. Retrieval stays valid because the needle is unique, but it is an
  easier task than the 653k and 1,046k rows, whose haystacks are purpose-built distinct text with
  200 of 200 sampled windows occurring exactly once.
- **The vLLM control is a different checkpoint** (`orcarouter/Qwen3.8-27B-Uncensored-FP8`, FP8
  weights, FP8 KV, speculation on) against NInfer's NVFP4 conversion of a different fine-tune. The
  two rows share a model family and a YaRN configuration, not weights. The row establishes the
  model-side retrieval ceiling under this YaRN configuration; it is not an engine-versus-engine
  comparison.
- **10/10 is not a demonstration of a high per-depth rate.** The 1M grid ran two samples per depth;
  for 10 successes in 10 trials the exact one-sided 95% Clopper-Pearson lower bound is 74%.
- **A needle just past 262k does not by itself prove YaRN works.** With the YaRN descriptors
  nulled, a 270k needle still resolved -- the model tolerates roughly 10% extrapolation past its
  trained ceiling unaided. The >262k legs of the real-weights YaRN test therefore establish
  *position addressability*, not YaRN quality. The 653k and 1,046k retrieval rows are what
  demonstrate YaRN's value, being at genuine multiples of the native window.

Retrieval, not recall: with an invented needle -- an invented place and an invented activity that
cannot be in any training set -- substituted at depth 50 of a 1,045,956-token prompt, the model
reproduced the sentence verbatim.

### Soak

All timings in this subsection were taken at the 400 W per-GPU cap (see
[Measured performance](#measured-performance) above); the soak was not re-run at 575 W.

Two consecutive passes decoded from a 949,885-token prompt to the exact 1,048,576-token ceiling:
98,692 generated tokens each, `finish reason context-capacity`, 52 minutes of wall clock per pass,
prefill 1,011.89 and 1,009.97 tok/s, decode 45.49 and 45.39 tok/s, per-device residency flat at
28,070 MiB (span 4 MiB), and the two passes' 98,692-token id streams and 380,672-byte reply texts
hash identically. No OOM, no NaN: ten logit-sanity windows sampled every 10k tokens kept top-1
share at 6.43-6.60% with no out-of-domain ids.

Greedy decoding degenerates on this prompt at this context length, by two distinct mechanisms, and
the soak stream is therefore stability and determinism evidence rather than a sample of 1M-context
generation quality. Without a speculative backend and with `--ignore-eos`, the model finishes an
answer turn, the suppressed end-of-turn token lets it answer again, and it settles into a
1,903-token turn cycle repeated 46 times byte for byte. With MTP3 the ~950k greedy stream contains
no end-of-turn token at all and still collapses, into a 187-token content-level loop whose first
repeat begins at generated index **1,201** -- that one is plain greedy degeneration, not an
`--ignore-eos` artifact. Sampled decoding at temperature 0.8 does not loop.

### YaRN reference and drift guard

NInfer's YaRN frequency correction is computed to match vLLM as deployed, including vLLM's
multimodal rotary path: correction range `(16, 24)` and an `mscale` of 1.13863 applied to `cos`/`sin`
over the 64 rotary dimensions of each 256-dimension head. The reference values are checked in at
`tests/core/data/yarn_ref_4x.json` (recorded against vLLM 0.25.1), and
`ninfer_qwen3_6_yarn_rope_drift_test` regenerates them from the installed vLLM and fails if either
the numbers or the vLLM version drift. Without a vLLM environment the test skips. To regenerate the
reference after a deliberate vLLM upgrade:

```bash
VLLM_LOGGING_LEVEL=WARNING "$NINFER_VLLM_PYTHON" tools/tp2/dump_yarn_ref.py \
  > tests/core/data/yarn_ref_4x.json
ctest --test-dir build -R ninfer_qwen3_6_yarn_rope_drift_test --output-on-failure
```

`NINFER_VLLM_PYTHON` points at the Python interpreter of the vLLM environment; the test falls back
to a known local path when the variable is unset, and `NINFER_YARN_REF_JSON` overrides the compared
file.

### Test environment variables

The dual-GPU and YaRN tests are opt-in the same way the rest of the artifact-gated suite is: each
reads an environment variable, falls back to a local default path, and skips (rather than fails)
when the resource is not present.

| Variable | Used by | Default |
|---|---|---|
| `NINFER_QWEN3_8_27B_NVFP4_WEIGHTS` | the sharded-materialization and TP2 real-Engine CTest routes | `/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer` |
| `NINFER_VLLM_PYTHON` | `ninfer_qwen3_6_yarn_rope_drift_test`, and `tools/tp2/dump_yarn_ref.py` above | `/home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python` |
| `NINFER_YARN_REF_JSON` | `ninfer_qwen3_6_yarn_rope_drift_test`, to compare against a scratch reference instead of the committed one | `tests/core/data/yarn_ref_4x.json` |

`scripts/tp1-regression.sh` takes the artifact as its first positional argument instead, and
`TP1_GPU` selects the physical GPU it runs on.

### Limitations

- **Vision works at `--tp 2` in this fork** (see [Vision on TP2](#vision-on-tp2)): the base fork
  ran the Vision encoder on the primary device only and rejected `--tp 2 --vision` at startup;
  this fork dual-replicates the tower and encodes per rank. YaRN is still rejected together with
  `--vision`, because the encoder ropes 2-D image-grid positions.
- **DFlash is unchanged and is rejected at `--tp 2`.** It remains a 35B-A3B text-only backend, and
  that target has no tensor-parallel path at all.
- **No NVLink, and no peer-to-peer on GeForce.** `cudaDeviceCanAccessPeer` reports 0 between two
  RTX 5090s, so the collectives are host-staged asynchronous copies over PCIe rather than direct
  peer copies. This is a measured property of the hardware, not a configuration choice. A decode
  token costs 128 reduces plus one logit all-gather; a 10 KiB reduce measures about 16 us, and
  under CUDA Graphs the whole collective set costs roughly 0.2 ms per token (both at the 400 W
  per-GPU cap).
- **MTP prefix reuse resets at `--tp 2`.** Resuming a prefix drives the MTP head from a retained
  target hidden state that only the primary device holds, so `--tp 2 --spec mtp` downgrades every
  reuse to a full prefill. The answer is unchanged and no request fails; the saving is lost, which
  matters for multi-turn conversation at long context.
- **MTP is output-equivalent up to near-tie argmax flips, not bit-identical.** A verify round
  evaluates the target model over `K+1` columns at once and an ordinary round over one, which
  selects different GEMM shapes; greedy MTP-on and MTP-off streams can therefore diverge on a
  near-tie token. Every committed token is still one the target model's own argmax selected. On the
  1,046k needles the MTP3 and MTP-off answers are identical token for token; on a 949,885-token
  soak stream they first differ at generated index 45. Measured like for like at 575 W -- same
  1,045,954-token prompt, same 512-token budget -- MTP3 decodes **2.18x** faster than MTP off.
- **1,048,576 tokens is a one-slot configuration.** `--max-concurrency` must be 1; concurrency at
  extended context requires coming down from the ceiling (roughly 500k for two slots at INT8 KV).
- **`--ignore-eos` is a diagnostic flag.** It exists for fixed-length soak and throughput work.
  Generation past the end-of-turn token is off-distribution and is not a product output. It is not
  the only route to a degenerate stream: greedy decoding at this context length also loops on its
  own content, with the flag off.
- **A full-ceiling prefill costs about 18 minutes** (17.9 at 575 W, 18.8 at the 400 W cap), and 1M
  decode runs at roughly 48 tok/s, or 100 tok/s with MTP3. Retrieval is not the binding constraint
  at 1M; latency is.
- The decode split policy was tuned at 262k. At 1M it is smooth rather than pathological, but it
  has not been swept at that length.
- **A 1M boot can fail on the CUDA Graph allowance, transiently.** One 1,048,576-token
  `--tp 2` boot aborted at startup with `CUDA Graph preparation consumed 44433408 bytes on device
  0, exceeding the planned per-device allowance of 20971520 bytes`. The identical command line had
  booted minutes earlier and booted again on the immediate retry, and successful boots of that same
  configuration report only 2.00-3.44 MiB of captured graph memory per device, so the 20 MiB
  per-device allowance is too tight against a capture pool that is not deterministic. The
  workaround is to retry; the fix is to raise the allowance or size it from a measured reservation.
  The failing log is kept at
  `eval/results/cross-engine-nvfp4/ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt`.
- **Three power conditions, and every number carries one.** The campaign ran at a **400 W** per-GPU
  cap, the publishable subset was re-measured at **575 W**, and the cross-engine comparison against
  vLLM was taken at **500 W**. Never quote a figure without its condition, and never compare figures
  across conditions.
- **What is not measured.** Concurrency C=2 was measured only at the 400 W cap, and the 653k tier
  was not re-measured at 575 W. The soak is greedy only; a seeded temperature > 0 soak has not been
  run. The cross-engine comparison has no vLLM MTP-off row -- speculative decoding is fixed at
  vLLM's launch and that restart was not made -- and the prefill-chunk difference behind vLLM's
  prefill lead (`--prefill-chunk 1024` against `--max-num-batched-tokens 16768`) was not swept on
  either engine. Independent FP64-oracle conformance for shard extents covers the FP8 A8 row
  extents only -- the NVFP4, Q4, Q5, W8 and vocabulary shard geometries are qualified pairwise
  against the `--tp 1` kernel on the whole weight, plus the model-level parity, retrieval and
  MTP-argmax evidence.

This section and [Performance](docs/performance.md) carry the headline figures with their
reproduction commands. The design decisions behind them -- the collective transport, the shard map,
the YaRN constants, and what each correctness gate actually proves -- are in
[Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md).

## Capabilities

All three registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

At `--tp 2` the Windows-TP2 fork's published validation covers image prompts (see
[Vision on TP2](#vision-on-tp2)); the multimodal list above is the upstream `--tp 1` contract,
where the 32,768 merged-token envelope applies instead of the tp2 per-item budget.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

## Current limits

- Only the six `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- Execution is specialized for the RTX 5090. One CUDA device is the default; the 27B execution
  package also runs on exactly two with `--tp 2 --devices A,B`, which is a capacity feature rather
  than scale-out.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving. Multi-GPU execution is exactly the two-device
  tensor-parallel width described above: one process, one resident model, no NVLink, no more than
  two devices.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit, or up to 1,048,576 tokens under `--rope yarn` on the 27B
  targets. `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

**Windows-TP2 fork additions:**

- [Windows peer mailbox transport (technical writeup)](docs/windows-peer-mailbox.md)
- [Benchmark results and methodology (this fork)](benchmarks/windows-tp2-benchmarks.md)
- [Vision-on-TP2 phase 3A research notes](research/phase3a-vision-architecture.md) — architecture
  audit, [memory map](research/phase3a-vision-memory.md),
  [TP2 design](research/phase3a-vision-tp2-design.md),
  [validation record](research/phase3a-vision-validation.md)
- [Provenance and attribution audit](docs/PROVENANCE.md)
- [Windows build notes (Russian)](docs/windows-tp2.md)

## Hardware tested (Windows-TP2 results)

| Component | Value |
|---|---|
| OS | Windows 11 x64, native (no WSL2/VM/Docker) |
| GPUs | 2 × NVIDIA GeForce RTX 5060 Ti 16 GB (`sm_120a`), same VBIOS |
| GPU0 slot | CPU-attached (Z690 PEG), PCIe 5.0 ×16 max — degrades to ×8 under load when an M.2 slot shares the CPU lanes |
| GPU1 slot | Z690 chipset, PCIe 3.0 ×4 (~3.16 GiB/s measured staging bandwidth) |
| CPU / RAM | Intel Core i3-12100F / 64 GB |
| Motherboard | MSI PRO Z690-A |
| Driver | 581.57 (WDDM) |
| Toolchain | MSVC 14.44 (VS 2022), CMake 4.0.2, Ninja, CUDA 13.1.80, vcpkg manifest deps |
| CUDA P2P | `cudaDeviceCanAccessPeer(0,1) == 0` (this machine/driver, GeForce/WDDM) |

The topology is unfavorable for tensor parallelism — one card on the CPU-attached slot, the second
behind the chipset at Gen3 ×4 — and CUDA peer access was not available between the cards under this
Windows 11 / WDDM driver. The measured result of the transport investigation is that for the small,
latency-critical TP2 messages of this engine the dominant cost was the staged protocol's fixed
synchronization overhead, not raw PCIe payload bandwidth, so useful TP2 decode throughput was still
reached after replacing the protocol (see
[benchmarks/windows-tp2-benchmarks.md](benchmarks/windows-tp2-benchmarks.md) §Communication
microbenchmarks). That conclusion is specific to this engine's message sizes and this machine; it is
not a claim that PCIe bandwidth is irrelevant in general.

## Building (native Windows)

Prerequisites: Visual Studio 2022 (C++ x64 tools), CMake ≥ 3.28, Ninja, a CUDA toolkit that can
compile for `sm_120a` (13.1 is known-good; the system CUDA 12.8 miscompiles one GQA decode kernel
with a shared-memory overflow — see [docs/windows-tp2.md](docs/windows-tp2.md)), and a vcpkg
checkout for the FFmpeg/libcurl/zlib manifest dependencies.

```bat
git clone --branch windows-tp2 https://github.com/ivanov84/ninfer-windows-tp2
cd ninfer-windows-tp2
git clone https://github.com/microsoft/vcpkg third_party\vcpkg
cmd /c scripts\build-windows.cmd all
:: or, with a non-PATH CUDA:
:: set NINFER_CUDA_BIN=C:\path\to\cuda\bin && cmd /c scripts\build-windows.cmd all
```

Artifacts land in `build-windows\apps\{ninfer.exe,ninfer-serve.exe}` (the CUDA runtime is
statically linked; FFmpeg/curl DLLs are resolved from the vcpkg install).

## Running (Windows-TP2 configuration)

```bat
build-windows\apps\ninfer.exe models\qwen3_8_27b_nvfp4.ninfer --tp 2 --devices 0,1 ^
  --max-context 8192 --kv-capacity auto ^
  --spec mtp --draft-tokens 4 --lm-head-draft ^
  --prompt "Explain tensor parallelism briefly." --max-new 512 ^
  --no-thinking --greedy --ignore-eos
```

`--tp 2 --devices 0,1` splits the model across both cards; `--spec mtp --draft-tokens N` selects
MTP speculative decoding; add `--vision` and a `--messages` multimodal JSON file (see
[Vision on TP2](#vision-on-tp2)) for image prompts; `NINFER_TP2_MAILBOX=0` in the environment
forces the staged transport (A/B testing). Diagnostics: `tools/tp2/mailbox_probe.cu` (mailbox vs
staged, exact-sum check), `tools/tp2/reduce_bench.cu` (per-collective latency),
`tools/tp2/graph_launch_probe.cu`.

## Benchmark methodology

All Windows-TP2 numbers in this fork were produced as follows, unless stated otherwise:

- single request, **greedy decoding** (temperature 0), **thinking disabled** (`--no-thinking`),
  `--ignore-eos`, fixed prompt ("Explain tensor parallelism briefly.", 19 tokens), generated
  tokens = 512 (1024/2048 for the long runs, labelled explicitly), `--max-context 8192`,
  `--kv-capacity auto`;
- tok/s = tokens **committed by the engine per decode-second** (from the engine's own
  end-of-run summary), not client-side wall-clock throughput and not server TTFT-inclusive
  figures — server/client timing must not be confused with these;
- MTP rows state their draft window (`--draft-tokens`) and acceptance rate; tok/s rises with
  acceptance, so MTP4@512 and MTP4@2048 are different workloads, not a single speed;
- each 512-token configuration was run at least twice; repeated runs agreed within ±0.5%.

Raw per-run data: [benchmarks/windows-tp2-final.csv](benchmarks/windows-tp2-final.csv) and
[benchmarks/windows-tp2-phase2c.csv](benchmarks/windows-tp2-phase2c.csv).

## Attribution

This fork stands on three layers of prior work, none of which it claims ownership of:

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the engine itself, including the Vision
  tower, MTP speculative decoding and serving routes (upstream, Apache-2.0);
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) (Wael Mansour) — the TP2 +
  YaRN fork this branch builds on: the `--tp 2` execution, the staged event-ordered allreduce that
  this fork optimizes, the cross-device CUDA graph bridge, and YaRN 1M-context scaling
  (base commit `6a355d5`);
- [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows) — the native Windows port
  whose compatibility layer was cherry-picked into this branch.

The work of this fork (the Windows-TP2 layer): the native Windows 11 TP2 bring-up and benchmark
campaigns, the peer-mailbox transport, vision-on-TP2 (dual-replicated tower, per-rank encode,
per-item pixel budget, MTP + image prefill fix, request-lane failure isolation), and the
`nvfp4-split` artifact support for the Ostfralla Qwen3.8-27B NVFP4 export. The full audit — which
files came from where, what was inspiration only, and what is external benchmark reference — is
[docs/PROVENANCE.md](docs/PROVENANCE.md); the Apache-2.0 §4(b) attribution lives in
[NOTICE](NOTICE).

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork's modifications are under the
same licence; see [NOTICE](NOTICE) for the attribution required by Apache-2.0 §4(b).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
