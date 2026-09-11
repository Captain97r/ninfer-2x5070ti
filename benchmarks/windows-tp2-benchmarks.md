# Windows-TP2 benchmark record (this fork)

Validated numbers for the peer-mailbox transport on the tested machine. Raw per-run data is in
[windows-tp2-final.csv](windows-tp2-final.csv) (Phase 2A/2B) and
[windows-tp2-phase2c.csv](windows-tp2-phase2c.csv) (Phase 2C sweep + long runs). The technical
explanation of the transport is [../docs/windows-peer-mailbox.md](../docs/windows-peer-mailbox.md);
the methodology common to every run is in the README's Benchmark methodology section.

## Machine

2 × RTX 5060 Ti 16 GB (`sm_120a`) · Windows 11 x64 native (WDDM, driver 581.57) ·
MSI PRO Z690-A · i3-12100F · 64 GB RAM · GPU0 on a CPU-attached slot (PCIe 5.0 ×8 — the RTX 5060
Ti is an ×8 card), GPU1 on the chipset (PCIe 3.0 ×4) · `cudaDeviceCanAccessPeer(0,1) == 0`.

Model: `qwen3_8_27b_nvfp4.ninfer` (Qwen3.8-27B NVFP4, SHA-256 verified against the published
artifact). Single request, greedy (temperature 0), `--no-thinking`, `--ignore-eos`,
`--max-context 8192`, `--kv-capacity auto`, CUDA graphs on, `--tp 2 --devices 0,1`.
tok/s = engine-committed decode speed from the end-of-run summary.

## Timeline of results

| Phase | Transport | Configuration | tok/s |
|---|---|---|---:|
| 2A (staged) | host-staged allreduce | MTP0, 512 tok | 16.37 |
| 2A (staged) | host-staged allreduce | MTP3, 512 tok | 32.58 |
| 2B (mailbox) | pinned-host mailbox | MTP0, 512 tok | 35.77 |
| 2B (mailbox) | pinned-host mailbox | MTP3, 512 tok | 63.19 |
| 2C (mailbox + tuning) | pinned-host mailbox | MTP0, 512 tok ×2 | 35.73 / 35.77 |
| 2C | pinned-host mailbox | MTP1, 512 tok | 57.71 |
| 2C | pinned-host mailbox | MTP2, 512 tok ×2 | 63.18 / 63.11 |
| 2C | pinned-host mailbox | MTP3, 512 tok ×2 | 66.71 / 66.86 |
| 2C **optimum** | pinned-host mailbox | **MTP4, 512 tok ×2** | **68.53 / 68.87** |
| 2C | pinned-host mailbox | MTP5, 512 tok (rejected) | 60.82 |
| 2C long | pinned-host mailbox | MTP4, 1024 tok ×2 | 70.74 / 70.57 |
| 2C long | pinned-host mailbox | **MTP4, 2048 tok** | **76.65** (acc 62.33%, 3.49 tok/round) |

Notes on the long runs: the 2048-token run rode a draft-acceptance rise (53.06% → 62.33%) as the
text accumulated repetitive markdown structures — the figure is workload-dependent and must not
be quoted as a general speed. Prompt-length sensitivity at 512 tok (MTP4): short 19-token prompt
68.65, medium ~100-token 66.97, long ~230-token 59.54 tok/s (KV growth costs ~13% on the long
prompt).

## MTP4 acceptance and round shape

| Run | acceptance | mean accepted/round | rounds/512 tok |
|---|---:|---:|---:|
| MTP4 @512 (×3 identical) | 53.06% | 3.12 | 164 |
| MTP4 @2048 | 62.33% | 3.49 | 586 (2048 tok) |

Determinism: repeated runs produced bit-identical greedy text and identical acceptance counts
(MTP4@512: 53.06% three times; 2048-token output continues the same sequence as the
512-token one; post-rebuild control run matched the base numbers exactly).

## Communication microbenchmarks

Per exchange pair, medians over paired kernels from Nsight Systems traces:

| Exchange size | dev0 / dev1 median | transport floor | lockstep excess |
|---|---:|---:|---:|
| 10 KiB (MTP0 shape, 16384 pairs) | 17.3 / 19.3 µs | 15.1 / 19.1 µs | ~0 |
| 40 KiB (MTP4 shape, 6300 pairs) | 59.3 / 65.3 µs | 16.8 / 19.5 µs | 42.5 / 45.9 µs |

Staged path for comparison: ~277 µs per 10 KiB reduction (reduce_bench, pre-mailbox). Staged
one-way bandwidth over the chipset link: 3.16 GiB/s (256 MiB transfers); small-transfer latency
~70 µs per 10 KiB copy. Pure payload time for 40 KiB at Gen3 ×4 is ≈13 µs — the transport floor
is flag round-trip + launch, not bandwidth.

## Round profile (MTP4, Nsight Systems, medians)

| Category | share of round |
|---|---:|
| NVFP4/W8 GEMM (memory-bound, ~95% of practical VRAM bandwidth) | ~60% |
| communication (mailbox exchanges) | ~18% |
| in-graph lockstep gaps | ~11% |
| host between graph replays | ~4% |
| lm_head | ~3% |
| GDN | ~2% |
| attention | <1% |

## External reference point (not a measurement of this fork)

`club-5060ti` community evidence for the same GPU pair, vLLM under Linux/Docker, 122k context,
FP8 KV: median decode 67.293 tok/s (`qwen38-27b-nvfp4-vllm-2x5060ti-122k.json`). A different
serving stack, context length and KV format — quoted for orientation only, not as a
controlled comparison, and not as evidence of any Windows-vs-Linux claim.

## Rejected / reverted experiments

- MTP5: acceptance falls to 40.52% — rejected.
- Poll backoff 100 → 400 ns (`kPeerPollNs`): 68.77/68.72 vs 68.53/68.87 baseline — within
  noise, **reverted**; the canonical tree carries `__nanosleep(100)`.
- Reduce-scatter / sequence parallelism, GDN fold into the graph, PDL for the draft chain:
  evaluated as future work, not implemented (documented in the internal phase-2C report; the
  rationale — memory-bound GEMM makes split-batch overlap counterproductive, and every
  allreduce sits on the critical path — is summarized in
  [../docs/windows-peer-mailbox.md](../docs/windows-peer-mailbox.md) §10–11).
