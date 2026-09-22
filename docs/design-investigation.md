# Qwen3.8-27B on two RTX 5070 Ti cards

Research updated 2026-09-22 against `main` commit `f07b80b2`. Native Windows,
one session near 100K context, TP2, the pinned mixed-NVFP4 artifact and MTP3
remain the product target. Linux portability is retained but has not been tested
locally. The research baseline changed only documentation and measurements.
Validated implementation stages are recorded below; weights, activation precision,
KV format, sampling and launch defaults remain fixed unless explicitly qualified.

## Assessment

Further hardware specialization is justified. Several FP4 and FP8 schedules still
explicitly inherit RTX 5090 choices, and the Windows TP2 adaptation retains
avoidable vocabulary communication and repeated descriptor setup. The strongest
first candidates preserve existing represented values: exact distributed draft
argmax, ownership-correct descriptor reuse, and removal of unnecessary duplicated
communication. Cold prompt processing needs its own transport and GEMM analysis.

There is no defensible percentage forecast for the complete engine yet. The
previous attention tuning qualified a 13-22% operator latency reduction, while
its two single whole-engine runs differed by only about 3.1% in TG. This is not
a statistically established application speedup. Quality qualification and direct performance measurements govern acceptance. A
working CUDA timeline is needed for reliable attribution of bottleneck fractions;
exact optimizations can still be tested directly while that tooling is repaired.

## Foundation and hardware

The original [Neroued/NInfer](https://github.com/Neroued/ninfer) targeted a single
RTX 5090. The [Windows TP2 fork](https://github.com/ivanov84/ninfer-windows-tp2)
was measured on two RTX 5060 Ti 16 GB cards; it incorporates
[wamansou's TP2 work](https://github.com/wamansou/ninfer-tp2-1m) and
[natpate's Windows port](https://github.com/natpate/ninfer-windows).
Our fork preserves that history, LICENSE and NOTICE. The exact base and artifact
are recorded in [engine-source.json](../engine-source.json) and
[config/model.json](../config/model.json). The local artifact has 21,492,695,040
bytes; a similarly named newer artifact is not an interchangeable reference.
Models remain under `C:\LLM`.

| Per GPU | RTX 5060 Ti 16 GB | RTX 5070 Ti | RTX 5090 |
|---|---:|---:|---:|
| CUDA cores | 4,608 | 8,960 | 21,760 |
| SMs | 36 | 70 | 170 |
| Nominal memory bandwidth | 448 GB/s | 896 GB/s | 1,792 GB/s |
| VRAM | 16 GB | 16 GB | 32 GB |
| Verified L2 here | Not established | 48 MiB | 96 MiB |

The core, bandwidth and VRAM figures come from [NVIDIA's comparison table](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/).
The [Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
supplies the SM architecture and cache details; 36 SMs follows from 4,608 cores
at 128 cores/SM. Our [GPU probe](../diagnostics/gpu_probe.json) independently
reports 70 SMs, 48 MiB L2, 64K registers/SM, 1,536 threads/SM, 100 KiB shared
memory/SM and 99 KiB opt-in shared memory/block.

The pair has about 1.94 times the SM count and twice the nominal memory bandwidth
of two 5060 Tis, with the same memory capacity. These ratios are not prospective
software speedups: the current engine already benefits from the faster hardware.
Together the cards match a 5090's nominal memory bandwidth, but have 140 rather
than 170 SMs and two separate memory pools.

The measured host is a Ryzen 7800X3D with 32 GB RAM. Its active links are Gen5
x8/x4, and CUDA peer access is unavailable in both directions under this Windows
WDDM configuration. [Hardware report](../diagnostics/hardware_report.json).
The x4 link matters even after choosing TP: it limits bulk traffic and affects
synchronization. Equal compute shards are appropriate for two identical cards;
a 2:1 shard split does not follow from the PCIe widths.

These are SM120 GeForce GPUs. B200/SM100 kernels using different Tensor Core and
memory mechanisms cannot be copied directly. NVIDIA's
[CUTLASS GeForce NVFP4 example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
and [SM120 functionality guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)
are relevant references for block-scaled MMA, tiles and persistent schedules.
They do not constitute a replacement for this engine's TP2 runtime, artifact
layout or numerical contracts.

## Measurements and their limits

The shipping configuration is TP2 devices 0/1, INT8-G64 KV, MTP3, optimized draft
head, CUDA Graphs, 1,024-token prefill chunks and 102,400-token capacity. Existing
validation includes the 11-check focused suite, matched-schedule prefix restoration
and seven image-grounding checks. See [validation](../diagnostics/validation.json),
[prefix evidence](../diagnostics/prefix-validation.json) and
[vision evidence](../diagnostics/vision-validation.json). These establish their
specific functional and numerical properties; they are not a broad quality score.

New unprofiled measurements use one discarded warmup and three measured repetitions.
Model loading is excluded, and TG counts committed output tokens. They use the
existing token-ID corpus; the 100K input cycles its 65,536-token source. Graphs
are primed before measurement. Vision is disabled in this benchmark, whereas the
normal server launcher enables it. The model and configuration were unchanged.

| Prompt tokens | Generated tokens timed | PP tokens/s, mean +/- SD | TG tokens/s, mean +/- SD | Draft acceptance |
|---|---:|---:|---:|---:|
| 8,192 | 256 | 3,053.95 +/- 1.06 | 194.72 +/- 0.21 | 97.95% |
| 100,000 | 512 | 2,244.01 +/- 0.20 | 168.42 +/- 0.26 | 93.55% |

Full configuration, each repetition and speculative counts are retained in
[the research measurement record](../diagnostics/optimization-research.json).
The reported spreads are standard deviations across three runs, not confidence
intervals. The very high draft acceptance is characteristic of this synthetic
input and generated continuation. These numbers do not establish coding/chat
speed or useful-answer quality. They are not directly comparable with the user's
90-100 tok/s llama.cpp Q6_K conversation.

The older [untuned](../diagnostics/baseline-100k.json) and
[tuned](../diagnostics/tuned-100k.json) single 100K/128-token runs reported
2,232 PP tok/s and 160.07 versus 165.07 TG tok/s. The new longer-generation run
is a new baseline, not another optimization result.

A bounded profiling attempt uncovered a tooling limitation. Installed Nsight
Systems 2026.1.3 explicitly reports that driver CUDA API version 13.4 is not
supported and falls back to its 13.3 libraries. Hardware/node capture exited with
`0xC0000409`; software/graph capture exited with `0xC0000005`. The latter exported
partial NVTX data but no CUDA kernel, runtime-API or memcpy event tables. Neither
trace is usable for bottleneck percentages. Ordinary unprofiled inference passed.
The version warning is direct evidence of an unsupported combination, not proof
of the precise crash cause. Use a matching Nsight release before selecting an
optimization from GPU/CPU timing attribution. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

## Correctness work that must precede tuning

### Windows descriptor lifetime: corrected

The audit found that [Nvfp4TmaDescriptorBlock](../src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu#L70)
allocated descriptors on its compute stream but freed them on the legacy default
stream. The engine's [nonblocking streams](../src/core/device.cu#L108) do not
implicitly order that free after the descriptor-consuming kernel.
[Allocation contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html),
[stream semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).

The helper now retains its owning stream and frees on that same stream. Allocation,
copy, kernel use and free are ordered without changing arithmetic or dispatch.
The A4 numerical test now uses a nonblocking stream and explicitly retires its
legacy-stream fixture setup. The sibling fused SwiGLU launcher was already correct.

Validation: affected Linear, LinearAdd and SwiGLU operator tests passed separately
on each physical GPU; the three TP2 projection/pipeline tests passed; Compute
Sanitizer memcheck with stream-ordered race tracking reported zero errors. A real
8K-prompt/256-token TP2+MTP3 inference check passed. These checks qualify the fix;
the original unsafe ordering was established by source inspection, without claiming
that the old build's output corruption was reproduced.

### Quality qualification: implemented checks and remaining limits

The legacy real-model MTP test now measures top1-minus-the-actually-emitted-token,
rejects non-finite valid logits and excludes padded vocabulary rows. A host-only
regression checks that a low-ranked emitted token cannot hide behind a small
top-two gap. Its unchanged `max(0.5, TP1 worst regret)` threshold is an uncalibrated
legacy structural screen. Cold-prefilling each growing prefix does not reproduce
decode KV/GDN state, and the full test requires more than 16 GB on one GPU. That
full-model test has not been run on this pair; it is not a same-state proof.

The speculative round test now compares joint accepted-count/terminal-token
frequencies against an independent FP64 formula. Both physical GPUs passed 49,152
draws each across batch widths 1, 2 and 8, full 248,077-token domain, poisoned
padding, penalties, repeated proposals, shortened rounds and zero/unit proposal
probabilities. Licensed tokens, committed counts, lengths, anchors, untouched
inputs and allocation guards are also checked. The prespecified multinomial
marginal Chernoff/KL criterion uses a union-bound family error of 1e-6 under
independent uniform draws; it is not fitted to the observed output. This qualifies
the sampler transition for represented logits, not the entire model state.

The fixed [response panel](../tests/data/quality-panel.json) covers 18 text, JSON,
tool, English/Russian/Chinese, reasoning, provided-history and image tasks. The
reference scores **16/18**: it answers the inventory arithmetic with 41 instead of
37, and adds prohibited indentation to a correct code expression. Both failures
also occur in the preserved original engine and with MTP disabled. The stream
lifetime fix matches all 18 original observable responses exactly. These controls
do not establish the quantized artifact's quality against an unquantized model.

[The panel runner](../tools/test_quality.py) reports task success separately from
exact observable parity. Existing failures remain failures; a complete failed
capture may be a regression reference but is never called an all-tasks-passing
baseline. Exact comparison retains content, reasoning, raw tool arguments, finish
reason and completion-token count, excluding random response/tool IDs. It is not
a token-ID comparison. Artifact/runtime settings and image bytes are fixed by
the comparison contract. The history cases do not prove a particular prefix
checkpoint path, and this short panel does not establish long-context quality.

On Windows, [the owned-server launcher](../tools/run_quality_windows.py) starts
the shipping TP2/MTP3/INT8/102400-context/vision configuration on loopback port
19080 and stops it after each panel. Use Python 3.11:

```powershell
python tools/run_quality_windows.py --label reference
python tools/run_quality_windows.py --label candidate --baseline build/quality/reference.json
```

Capture currently returns exit 1 for the two real task failures. Comparison
returns success only for complete exact parity without new task failures, while
retaining `all_tasks_passed: false`. The server used for ordinary OMP requests
continues to use port 8000.

The documented sampler keeps at most 20 candidates. Normal Qwen3.8 presets use
20, and uncustomized OMP omits sampling overrides. Explicit top_k=0 or values
above 20 nevertheless get accepted by the HTTP layer and capped internally;
greedy also ignores penalties. Preserve the default baseline, and either reject
unsupported overrides clearly or implement their advertised semantics before
comparing those settings with another engine. See
[sampler contract](../include/ninfer/ops/sampling.h),
[Qwen presets](../src/targets/qwen3_6_27b/impl/package.cpp) and
[OMP's parameter omission behavior](https://github.com/can1357/oh-my-pi/blob/v18.2.8/docs/settings.md#sampling).

## Generation opportunities

### 1. Exact distributed draft argmax: implemented

The optimized draft head now uses exact rank-local reduction, a one-way
16-byte candidate exchange, and a global merge before the unchanged token mapping.
This replaces gathering both full 131,072-row draft vectors. Captured execution
uses an owned mailbox slot; eager execution and unsupported mailbox envelopes
use an event-ordered candidate transfer. No quantization, projection, target
verification or sampling arithmetic changed.

Independent exact tests passed in both physical GPU orders, covering non-finite
inputs, ties, padding, remapping, changed replays, scratch reuse and fallback
transports. All 18 saved response observables remain identical, including the two
known task failures. Repeated complete-inference measurements improved synthetic
TG by 2.55% at 8K and 2.46% at 100K, with matching speculative counts. The
[performance record](performance.md#local-dual-5070-ti-exact-draft-selection)
separates those results from the larger isolated selection speedup.

### 2. Avoid redundant target-logit communication

[Target verification](../src/targets/qwen3_6/impl/runtime/text_context_impl.h#L2733)
also gathers full logits onto both cards and repeats selection/acceptance. The
padded target vocabulary has 248,320 rows, while valid token-domain handling must
retain its 248,077 limit.

Greedy verification can use an exact distributed argmax. Stochastic verification
requires target probabilities: a coherent first alternative gathers onto rank 0,
performs acceptance there, then distributes licensed tokens and acceptance/state-selection
metadata. Stop/cancel retirement determines the committed prefix afterward. Penalty histories, RNG domains, accepted counts and both ranks' GDN/MTP
frontiers must stay consistent. Replacing target verification with a draft shortlist
would change semantics and is not part of this proposal.

From the current schedule, a steady MTP3 round performs 137 hidden allreduces
(131 at 40 KiB and six at 10 KiB), four target-logit gathers and three draft-logit
gathers. The vocabulary gathers add 1,386,496 bytes in each direction per round,
before host-staging legs. These counts come from the
[MTP schedule](../src/targets/qwen3_6/impl/runtime/mtp_impl.h#L234) and
[collectives](../src/ops/common/allreduce.cu#L320); they are not measured timing
shares. Ordinary decoding has 128 hidden allreduces and one target-logit gather
per token. Do not apply eager-allreduce microbenchmark latency to captured
mailbox collectives: their execution paths differ.

### 3. Retune actual 70-SM FP4 and FP8 shapes

[NVFP4 schedules](../src/ops/linear/nvfp4/nvfp4_config.h#L253) explicitly name
RTX 5090 cold-cache winners; the TP2 specializations inherit parent schedules.
[FP8 schedules](../src/ops/linear/fp8/fp8_config.h#L424) have the same issue.
Search warps/CTA, rows/warp, token tiling, activation staging, register pressure,
cache policy and block order on actual TP2 T=1 and T=4 shapes. Keep the existing
activation formats and reduction contracts fixed initially.

FP8 is important here: this artifact uses it for attention/GDN projections,
embeddings, the full head and its final eight MLP layers. An optimization effort
confined to nominally FP4 matrices would miss substantial work. Further attention
retuning should follow its measured fraction of full inference; isolated kernel
speedups cannot be added together into a TG forecast.

### 4. MTP windows require quality qualification

The hidden-reduction count grows as `128 + 3K` in a normal K-draft round, and each
extra draft adds sequential work. Optimize committed tokens divided by total
round time, not acceptance rate alone. Measure coding, prose, reasoning and long
histories; the current synthetic 94-98% acceptance is not representative evidence.

Crucially, [NVFP4 SwiGLU](../src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp#L40)
uses A16 through width four and W4A4 from width five. MTP3 verifies four tokens;
MTP4 verifies five. Increasing the draft count therefore adds target activation
quantization. Either hold the precision route fixed for the scheduling experiment
or qualify the changed route separately. The precision-changing variant is outside
the initial quality-preserving track; finite task tests cannot prove it universally
lossless. Leave MTP3 as the baseline meanwhile.

## Prompt-processing opportunities

Each of 64 target layers has two hidden reductions. A normal 1,024-token chunk
adds the MTP stem reduction, giving about 129 exchanges of 10 MiB per rank.
Across a 100K prompt this represents roughly 132 GB of logical hidden data in
each direction, before counting the host-staging legs. This large volume explains
why PCIe topology remains relevant even though the weights stay on the GPUs.

The eager [bulk reduction](../src/ops/common/allreduce.cu#L292) issues event
records, waits, two cross-device copies and two combine kernels. Prefill retires
both streams at chunk boundaries. Determine how much time belongs to transfers,
GPU work and host submission before choosing between these distinct changes:

- **Descriptor reuse:** both Windows TMA wrappers allocate/copy/free descriptors
  on each call. Preplanned per-Program, per-stream storage can remove allocation
  work without changing arithmetic, after the lifetime issue above is fixed.
- **Chunk and tile tuning:** evaluate 512/1,024/2,048/4,096 if memory admits it.
  Larger chunks reduce invocation counts but do not remove the total hidden
  payload. The current fused NVFP4 SwiGLU TMA route is selected only at exactly
  1,024 tokens; other sizes can lose fusion. Retune the route and tiles together,
  including workspace and numerical effects. A larger CLI number is not evidence
  of a better schedule.
- **Fixed-chunk prefill graphs:** appropriate if CPU submission gaps dominate.
  They need stable token-ingress buffers, descriptors and rank-local workspace,
  plus dynamic KV visibility. Preserve the large-payload staged transport; the
  tiny decode mailbox is not automatically suitable for 10 MiB exchanges.
- **Tiled transfer/compute overlap:** appropriate if communication is on the
  critical path and the dependencies allow overlap. Preserve exact BF16 sums and
  explicit stream lifetimes. Compressing hidden traffic to FP8/INT8 adds a new
  approximation and is excluded from the quality-preserving implementation track.
  Finite evaluation cannot establish that additional lossy compression is lossless.

An example occupancy question is the current down-projection grid at T1024:
40 times four tiles gives 160 CTAs over 70 SMs. Different tile sizes or staging
may improve wave utilization, but register/shared-memory pressure can reverse
that result. The SM120 CUTLASS examples are useful comparison implementations,
provided scale layout, activation quantization and epilogue semantics match.

## Session latency and alternatives

Text suffix prefix reuse and both-rank checkpoint restoration already work.
[Exact zero-suffix and multimodal reuse still reset](../src/targets/qwen3_6/impl/runtime/request_plan_impl.h#L227).
Supporting those cases could avoid an entire repeated long prefill when the
request pattern qualifies. Measure time to first token on real multi-turn requests;
a prefix hit is not an improvement in cold-prefill tok/s. Correctness requires
matching image identity/positions, token history, both GDN shards and MTP state.

Keep Linux as a comparison environment, not an assumed cure. Re-measure peer
access, link bandwidth and communication there. This host's Windows result does
not prove Linux enables P2P on these GeForce cards. NCCL and unrelated distributed
engines are not drop-in replacements for the native Windows runtime.

Likewise, a GPU-controlled decode loop is not a small graph edit: this TP2 engine
uses a cross-device graph, whereas CUDA conditional graph bodies and device-launched
graphs have single-device requirements. [CUDA Graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements).
Replacing the speculation backend or adopting a newer artifact is a separate
runtime, memory and quality project; prioritize changes to the present model first.

## Quality acceptance plan

There are three separate questions: whether the chosen quantized artifact is
good enough, whether TP2/MTP executes that artifact correctly, and whether a new
optimization degrades the accepted baseline. Existing NVFP4/FP8 weights, INT8
attention and draft-head approximations predate this investigation. None should
be silently changed to obtain a more attractive throughput number.

The MTP code uses full target verification. Greedy proposals are checked against
target argmax; stochastic one-hot proposals are accepted with target probability,
with residual sampling after rejection and a bonus token after full acceptance.
The [speculative decoding result](https://proceedings.mlr.press/v202/leviathan23a.html)
applies only if target distributions and persistent state are correct. Different
verification arithmetic can violate that premise. Stochastic MTP and ordinary
decode need not generate the same sequence for the same seed because their random
number purposes differ.

Use these acceptance gates before adopting a candidate:

1. **Freeze the references.** Keep `f07b80b2` plus its exact artifact/configuration
   as the regression reference; use the same artifact under TP2 with MTP disabled
   as the speculation reference. Run them sequentially. Keep tokenizer/template,
   prompt token IDs, prefill history/chunks, KV profile, thinking mode, sampler,
   output budget and image preprocessing fixed.
2. **Check numerical/state behavior directly.** Exact transfers, cache copies,
   unchanged-schedule replay and argmax rewrites must match exactly. Numerical
   kernels need independent FP32/FP64 oracles over their represented inputs and
   production shapes, including both GPUs and broad graph envelopes. Compare
   same-state target logits, emitted-token regret, log-probability changes and
   distributions; cold-prefilling every growing prefix is not a same-state test.
3. **Test speculative sampling distributions.** Use nondegenerate target
   probabilities, many seeds and an independent CPU reference. Cover each
   rejection position, zero/intermediate/unit draft probability, all-accepted
   bonus, provisional penalties, shortened final rounds and commit counts.
4. **Evaluate useful work.** Reuse the [existing evaluation framework](../eval/README.md)
   with concurrency one and budgets that fit 102,400 context. Cover executable
   code tests, tool/JSON validity, instruction following, English/Russian/Chinese,
   reasoning and factual tasks, 4K/32K/100K retrieval with distractors and absent-key
   controls, multi-turn rewind/append, and images. [EvalPlus](https://github.com/evalplus/evalplus)
   and [RULER](https://github.com/NVIDIA/RULER) are suitable sources of stronger
   code and long-context cases. Do not execute arbitrary model-generated programs
   outside an isolated test harness.
5. **Accept performance only with quality.** No regression in deterministic
   must-pass cases; numerical thresholds must follow the contract, not be relaxed
   until a candidate passes. For task scores, use paired cases/multiple seeds and
   uncertainty intervals; an inconclusive small suite does not prove equivalence.
   Report regressions by category, including unfinished answers, repetition,
   malformed tools and lost long-context facts. Measure PP, TTFT, committed TG,
   acceptance, memory and time to a correct completed answer. Average throughput
   must not hide slower common workloads or lower task success.

An initial practical task panel can use tens of fixed prompts spanning those
categories; it is a regression screen. High-confidence claims require broader
paired evaluation, especially for a changed precision route. Existing upstream
5090 benchmark/accuracy scores cannot substitute for local TP2 validation.

BF16-KV comparisons at short contexts are feasible on the pair; a full 100K
BF16 allocation may exceed available VRAM. INT8 attention also quantizes Q to
Q8-G64, so that comparison changes the attention compute profile as well as the
stored cache. See [attention contract](../include/ninfer/ops/gqa_attention.h#L44).
A 27B BF16 checkpoint is not a prerequisite for each local regression. Saved
reference logits and independently decoded operator inputs can establish useful
checks without fitting the entire BF16 model on one GPU. The current Python
reference binder accepts Qwen3.6 groupwise-int, not this Qwen3.8 NVFP4 artifact,
and would need explicit codec/binding adaptation before serving as its reference.

## Implementation order

1. Resolve the descriptor lifetime and strengthen same-state TP2/MTP and sampling
   qualification. Establish a representative quality/performance panel.
2. Implement exact distributed draft argmax and measure its direct effect, then
   qualify greedy target selection or rank-0 stochastic acceptance if useful.
   Restore supported CUDA timeline profiling for bottleneck attribution in parallel.
3. Remove descriptor churn and tune prefill chunks, fusion and bulk transport
   according to the timeline. Retune the important FP8/FP4 shard kernels on 70 SMs.
4. Qualify broader prefix reuse, MTP-window/precision experiments and Linux.

There is concrete room to investigate, but no measured basis yet for promising
200/250/300 tok/s on real coding sessions or a fixed PP multiplier. A speedup is
accepted only after it preserves the required observable behavior and useful-output
quality within the stated validation scope.
