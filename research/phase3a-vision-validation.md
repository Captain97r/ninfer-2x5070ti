# Phase 3A-4/3A-5/3A-6 - Vision TP2 validation results (Windows 11, 2x RTX 5060 Ti)

Machine: 2x RTX 5060 Ti 16 GB (`sm_120a`), Windows 11 x64 native (WDDM, driver 581.57),
`cudaDeviceCanAccessPeer(0,1) == 0`, no NCCL, no WSL2. Artifact: `Qwen3.8-27b-nvfp4.ninfer`
(20.71 GiB weights H2D, 992 tensors). All runs: `--tp 2 --devices 0,1`, greedy, CUDA graphs on.

## 1. Startup crash: root cause and fix

**Symptom**: `--vision --tp 2` aborted during engine construction at `ProgramImplCore`
(`program_impl.h:455`, `cudaMemsetAsync(io.rope_delta...)`) with
`cudaErrorInvalidValue: invalid argument`, exit `-1073740791`. Text-only tp2 never crashed.

**Root cause** (confirmed by a temporary probe printing `cudaGetDevice` at the crash site):
`RequestMemory::Impl` (`src/runtime/engine/request_memory.cpp`) called
`cudaSetDevice(1)` to build rank 1's transient arena while rank 0's setup was the current
device and **never restored it**. The subsequent rank-0 `Program` constructor then allocated
its persistent span **on device 1**; every rank-0 use of that span (starting with the first
`cudaMemsetAsync` on the rank-0 stream) was invalid. In text-only runs the request arena
capacity is zero, so the buggy `cudaSetDevice` never ran - which is exactly why the crash
was vision-only and looked like a vision problem while being a general device-scope leak.

**Fix** (commit `2cf050b`): the arena saves the previous device, allocates, and restores it
(exception-safe), and the teardown does the same. Text tp2 is untouched (capacity 0), and
the discipline is local to `RequestMemory`.

## 2. Functional verification (Level 2/3)

`scripts/phase3a-vision-verify.cmd` - two images, `--max-new 128`, greedy, thinking on:

| image | exit | vision | text prefill | prefill speed | decode | answer |
|---|---:|---:|---:|---:|---:|---|
| `visual_chart.png` | 0 | 0.048 s | 0.568 s / 468 tok | 824.5 tok/s | 35.65 tok/s, 122 tok | `NINFER VISION 731；3；左侧` |
| `natural_scene.png` | 0 | 0.048 s | 0.567 s / 457 tok | 806.3 tok/s | 35.66 tok/s, 128 tok | thinking still running at limit |

Grounding checks (streamed reasoning on stderr):

- chart: the model reads the exact chart content - title "NINFER VISION 731", the "COUNT"
  axis label, "three red circles", "POSITION" - and the strict-format prompt
  (title; count; left-or-right) is answered correctly: `NINFER VISION 731；3；左侧`.
- natural: the model reads the on-image caption "A SUNNY HOUSE BY THE MOUNTAINS", the
  mailbox number "24", sun position, mountains/grass/sky - a coherent structured
  description; 128 tokens only ran out mid-thinking (the model is verbose in thinking mode),
  and `--max-new 320` continues the same coherent analysis.

Determinism: every rerun with identical parameters produced **bit-identical** stdout
(SHA-256 compared across runs), matching the established tp2 mailbox determinism record.

Vision does not participate in decode: the vision phase runs once at prefill (0.048 s) and
decode speed equals the text-only tp2 speed (35.65-35.73 tok/s MTP0).

## 3. Text TP2 regression (canonical methodology)

19-token prompt `Explain tensor parallelism briefly.`, `--no-thinking --ignore-eos`,
`--max-context 8192 --kv-capacity auto`, single request, engine-committed decode tok/s,
run on the final committed tree (after the fix and the orphan-experiment revert):

| config | baseline (2C) | this tree | verdict |
|---|---:|---:|---|
| MTP0 @512 | 35.73-35.77 | **35.83** | within noise |
| MTP3 @512 | 66.71-66.86 | **66.68** | within noise |
| MTP4 @512 | 68.53-68.87 | **68.86-68.87** | identical; acceptance 53.06% bit-matches baseline |

No text-path code was changed by the fix (the arena path with capacity 0 is byte-identical
in text mode).

## 4. Memory (3A-5)

Engine-reported per-GPU (rank 0 shown; rank 1 mirrors it plus its own 282 MiB vision copy):

| component | per GPU |
|---|---:|
| text weights (split) | 10.36 GiB |
| KV pool (64 MiB payload, auto @8k) | 64 MiB |
| gdn state | 146.8 MiB |
| sequence + workspace + graphs | ~408 MiB |
| vision weights (replicated by design, architecture E) | 282 MiB |
| runtime reservation | 445.6 MiB |
| planned device total | ~10.8 GiB |
| free after startup | 3.94 GiB |

`nvidia-smi` before/after the heaviest vision run: 700 MiB / 0 MiB residual - clean teardown,
no leak across the twelve full load/encode/prefill/decode cycles of this session.

## 5. Performance summary (3A-6)

- vision encode: **0.047-0.048 s** per image (both test images) - one-shot per request,
  paid at prefill, never during decode.
- vision -> text handoff: included in the 0.567-0.568 s text prefill (806-824 tok/s over
  457-468 prompt tokens including image embeddings).
- TTFT for a 468-token multimodal prompt: ~0.62 s.
- decode: unchanged from the text-only tp2 path (35.65-35.73 MTP0, 66.68 MTP3, 68.87 MTP4).

## 6. Known limitations

- The full-model tp1 oracle does not fit on one 16 GB GPU (20.71 GiB of weights), so the
  correctness argument is: (a) per-rank encode is the unmodified tp1 vision code with the
  same weights, (b) bit-identical deterministic reruns, (c) exact grounding of on-image
  text/counts/positions in both directions (chart and natural photo), (d) the text path
  bit-matches its pre-vision benchmarks. A torch reference component dump
  (`tools/reference` has `--activation-dump`) remains future work when a torch
  environment is provisioned on this machine.
- Thinking-mode image answers need a generous `--max-new` (the model reasons at length);
  with `--no-thinking` the strict-format chart answer is immediate.
- The tp2 vision envelope caps merged items at 2048 (documented in the design doc);
  oversized video prompts are rejected loudly at plan time.

## 7. Reproduction

```
build-windows-131\apps\ninfer.exe Qwen3.8-27b-nvfp4.ninfer --tp 2 --devices 0,1 ^
  --vision --greedy --no-thinking --max-new 64 ^
  --messages examples\cli\messages\image_chart.json
scripts\phase3a-vision-verify.cmd
```
