# Provenance and attribution audit

This document is the audit of where every layer of this fork comes from, performed before the
fork was published and before any upstream communication. It distinguishes **direct code
derivation**, **cherry-picked code**, **inspiration**, and **external benchmark reference**, and
it records the license position of each source. It is the authoritative attribution record; the
[NOTICE](../NOTICE) file carries the Apache-2.0 §4(b) subset.

## Repository graph

```
Neroued/ninfer                (upstream engine, Apache-2.0, master @ 863aa8a*)
   └── wamansou/ninfer-tp2-1m (TP2 + YaRN 1M fork, feaf4dd + 6a355d5)
          └── THIS FORK       (windows-tp2 branch: 6a355d5 + c0a64bb + docs
                               + vision-TP2 / nvfp4-split layer e67bac9..7b0852f)
                   │
                   └── cherry-picks from natpate/ninfer-windows
                       (native Windows port, itself a fork of Neroued/ninfer)

natpate/ninfer-windows        (Windows port; its README credits a "ninfer-3090 fork"
                              as the origin of its compatibility layer)

* upstream master has advanced past the feaf4dd base; this fork does not track it.
```

## Layer-by-layer

### 1. Upstream engine — `Neroued/ninfer` — **derived code (whole tree)**

- Everything except the layers listed below is upstream NInfer, Apache-2.0, unchanged from the
  `feaf4dd` base commit the TP2 fork branched from.
- Upstream is a **single-GPU** engine by design (README: "Maximum single-GPU inference
  performance"; no multi-GPU). It contains **no tensor-parallel code**; therefore none of the
  transport work in this fork can be a direct contribution to upstream as-is.

### 2. TP2 + YaRN base — `wamansou/ninfer-tp2-1m` @ `6a355d5` — **derived code (fork parent)**

- The `--tp 2` execution, the staged event-ordered allreduce (the code this fork optimizes),
  `PeerEvents`, the cross-device CUDA graph bridge, and the YaRN 1M-context scaling are the work
  of **Wael Mansour** (wamansou), on top of upstream `feaf4dd`.
- The staged fallback path itself (`allreduce.cu` before this branch) is *also* wamansou's; this
  fork modifies it (adds mailbox selection) but did not write it.
- Basis for attribution: the fork's NOTICE, its maintainer/tp2-yarn-1m.md design doc, and the
  git history (`6a355d5` is the direct parent of this branch's work).

### 3. Native Windows port — `natpate/ninfer-windows` — **cherry-picked code**

This branch's Windows compatibility layer was cherry-picked from natpate's fork and squashed
into `c0a64bb`; the squash destroyed the per-commit `Co-Authored-By` trailers, so the mapping is
restored here (verified against the reflog of the squashed intermediate commits and byte-for-byte
file comparison with natpate's master):

| Squashed local commit | Source (natpate/ninfer-windows) | Content |
|---|---|---|
| `b16bdef` | `8e7f596` | CMake/vcpkg native Windows build support |
| `a8f44af` | `965c67f` | `CreateFileW`/`MapViewOfFile` artifact reader, load progress, console log |
| `ee9e1ea` | `0785bd3` | Winsock media acquisition + request-log test port |
| `1930db1` | `9bc46ce` | Platform-routed library dependencies |
| (adapted) | `48035f7` + `d7cc655` + `b408783` | MSVC-safe TMA descriptors and plan move bodies |

Byte-identical to natpate's master at audit time: `src/artifact/reader.cpp`,
`src/product/load_progress/load_progress.cpp`, `src/serve/console_log.cpp`, `vcpkg.json`,
`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh`, `nvfp4_linear_swiglu_w4a4_tma.{cu,cuh}`.
`src/serve/request_log.cpp` and `CMakeLists.txt` diverge (this tree carries wamansou's TP2
additions on top).

natpate's own README credits a **"ninfer-3090 fork"** as the origin of their compatibility
layer — a third-hand origin we do not independently verify; the two-hop chain
(ninfer-3090 → natpate → here) is recorded as natpate states it.

### 4. Peer-mailbox transport — **original work of this fork** (`c0a64bb`)

- `include/ninfer/ops/peer_mailbox.h`, `src/ops/common/peer_mailbox.cu`,
  `src/ops/kernel/peer_exchange.cuh` — the mailbox slab, exchange kernels, and protocol;
- the mailbox selection in `src/ops/common/allreduce.cu` (staged path preserved as fallback);
- graph integration (`graph_impl.h` hang-guard + flag reset, `program{,_impl}.h` install and
  slot sizing), MTP-related cleanup, CUDA 12.8/MSVC build fixes (12.8 ptxas shared-memory
  overflow workaround, LNK2005 cudart flavor fix in tests);
- diagnostics: `tools/tp2/{mailbox_probe,reduce_bench,graph_launch_probe}.cu` and the
  `tools/tp2/*.py` profile analysis scripts (the latter were authored with hardcoded local
  paths, since neutralized in this branch);
- the benchmark campaigns in `benchmarks/`.

Not copied from any external repository; no public implementation of this transport was
referenced.

### 4b. Vision-on-TP2 and the split-storage artifact profile — **original work of this fork**
(`e67bac9`, `2c65596`, `bb90399`, `7f96fb2`, `7b0852f`)

- **Vision at `--tp 2`**: the base fork rejected `--tp 2 --vision` at startup. This layer
  dual-binds the vision tower per rank (`bindings.cpp` — every `vision/` object on a Replicated
  placement, ~282 MiB per GPU), runs the unmodified tp1 vision encode on each rank, publishes the
  embeddings to the peer through the existing PeerMailbox publish path, and consumes them in the
  text TP2 prefill (`layouts_impl.h`, `text_context_impl.h`); it sizes the tp2 vision workspace
  for one item capped at 16,384 merged tokens and clamps the frontend preprocessor budgets so
  `smart_resize` downscales oversized media (`prepared_prompt.h`, `frontend.cpp`), fixes the MTP
  draft-stage rope-position gather over the vision-merged axis, and isolates request-scoped
  prefill failures to the failing lane (`concurrent_executor.h`). Design/audit record:
  `research/phase3a-vision-{architecture,memory,tp2-design,validation}.md`; verification harness:
  `scripts/phase3a-vision-verify.cmd`.
- **`qwen3.8-27b / nvfp4-split` artifact identity** (`package.cpp`, plus BF16 column-shard
  siblings of the fused attention input projection kernels): the split-storage Qwen3.8-27B NVFP4
  export published by Ostfralla (separate GDN a/b projections, BF16 early attention input
  projections, W8G32 vocabulary planes) needs a binding distinct from the neroued mixed
  FP8/NVFP4 `nvfp4` schema already registered upstream, so the schema is registered under its own
  identity. The tested artifact file is the published Ostfralla file with the six manifest bytes
  of the identity string relabeled; the weight payload is byte-identical (manifest object tables
  compared, payload ranges sampled at identical absolute offsets across the whole file, plus the
  file tail).

### 5. Inspiration only — **no code reused**

- `syv-ai/qwen38-27b-rtx3090` — a single-RTX-3090 vLLM serving recipe for the same model family
  that motivated the original experiment question ("can 2 × 5060 Ti serve Qwen3.8-27B?"). No
  code from this repository is present in this fork.
- `5p00kyy/club-5060ti` — a community 2 × RTX 5060 Ti benchmark/presets repository. Its hardware
  lane organization and its community result template shaped how this fork documents its own
  hardware and benchmark methodology. No code from this repository is present in this fork.

### 6. External benchmark reference — **data citation only**

- `club-5060ti` evidence file `data/evidence/qwen38-27b-nvfp4-vllm-2x5060ti-122k.json`
  (median decode 67.293 tok/s, vLLM, Linux/Docker, 122k context, FP8 KV): quoted in this fork's
  benchmark files as an external orientation point. **It is not a measurement of this fork**, it
  is a different serving stack on similar hardware, and the fork's documents label it as such
  wherever it appears.
- llama.cpp (GGUF Q6_K, layer-split `--split-mode layer --tensor-split 54,46`) was run locally
  during Phase 2A as a cross-engine sanity check (17.49 tok/s MTP0 / 32.30 tok/s MTP3). It is
  mentioned in the internal phase reports; no llama.cpp code is part of this fork.

## Commit accounting

| Commit | Author | Content |
|---|---|---|
| `feaf4dd` (upstream) | Neroued & contributors | engine base |
| `6a355d5` | Wael Mansour | TP2 + YaRN 1M fork (branch parent of this work) |
| `c0a64bb` | Dmitriy Ivanov | Windows port (cherry-picked from natpate, attributed above) + original peer-mailbox transport + diagnostics |
| (this branch, docs commit) | Dmitriy Ivanov | README/docs/benchmarks/NOTICE/provenance + local-path cleanup |
| `e67bac9` | Dmitriy Ivanov | vision at `--tp 2`: dual-replicated tower + per-rank encode (phase 3A) |
| `2c65596` | Dmitriy Ivanov | Windows test infra (LF-pinned fixtures, cudart flavor for tp2 tests) |
| `bb90399` | Dmitriy Ivanov | BF16 TP2 column-shard attention input kernels + `qwen3.8-27b/nvfp4-split` profile |
| `7f96fb2` | Dmitriy Ivanov | BF16 wrapper split-path fix; verified Ostfralla `nvfp4-split` loads and serves at ctx 131072 |
| `7b0852f` | Dmitriy Ivanov | vision TP2 fixes (release r4): pixel-budget clamp/downscale, MTP+vision rope fix, request-lane failure recovery |

The `c0a64bb` commit message itself credits the Windows layer to natpate's fork; the per-file
mapping above supersedes the squash in precision.

## License observations

- All five repositories in the graph (`Neroued/ninfer`, `wamansou/ninfer-tp2-1m`,
  `natpate/ninfer-windows`, and the two inspiration repos) are **Apache-2.0** licensed at audit
  time. Vendored dependencies keep their own licenses under `third_party/`.
- Apache-2.0 §4 requires (a) retaining the LICENSE, (b) stating changes made, (c) retaining
  notices/attribution. This fork: keeps upstream LICENSE verbatim; states changes in the README
  banner, this file, and the NOTICE; carries no upstream NOTICES file conflict (upstream ships
  none — its attribution is README-based, which the fork banner preserves).
- The model artifacts referenced (`qwen3_8_27b_nvfp4.ninfer` etc.) are **not** redistributed by
  this fork — no model files, weights, or build outputs are committed.

## Publishing hygiene performed before this fork went public

- scanned the tree for credentials and API keys: none found in tracked files;
- removed hardcoded personal machine paths (`D:\AI_envs\...`) from the analysis scripts;
- excluded from the repo: raw profile sqlite files, personal benchmark logs, model files, build
  artifacts, and the internal Russian-language phase reports with mojibake and machine paths
  (their validated numbers are republished in English in
  [benchmarks/windows-tp2-benchmarks.md](../benchmarks/windows-tp2-benchmarks.md));
- the committed benchmark CSVs were checked to contain only relative log references.
