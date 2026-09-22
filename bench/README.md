# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central public Op contracts while leaving
implementation selection behind those contracts. Target benchmarks measure Program/model
composition. Correctness and model parity live outside this directory; development rules are in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target ninfer_bench
```

## TP2 draft selection comparison

`ninfer_argmax_row_parallel_bench` compares the former full-logit gather, argmax and remap
with exact distributed argmax and remap, using both mailbox and staged transport. It uses
65,536 BF16 rows per rank, resident input logits, CUDA Graphs, ten warmups and 31 interleaved
samples. Origin-stream events enclose each complete cross-device graph; input uploads,
oracle checks and model projection are outside timing. Results describe selection latency,
not an end-to-end generation speedup.

```bash
cmake --build build -j --target ninfer_argmax_row_parallel_bench
./build/bench/ninfer_argmax_row_parallel_bench --cols 1
./build/bench/ninfer_argmax_row_parallel_bench --cols 8 --reverse-devices
```

The executable checks every compared route against the shared scalar argmax oracle and
the subsequent token-ID map. The matching `ninfer_argmax_row_parallel_test` covers nonfinite
values, ties, padded domains, repeated replays, arena reuse and mailbox fallback cases.
Run it again with `--reverse-devices` to exchange primary/peer roles.

## Product benchmark

### Short-prompt Windows serving comparison

`tools/bench/compare_serve_windows.py` compares a preserved server binary with a
candidate using the unchanged `scenario_code_python`, `scenario_story_en_mystery`,
and `scenario_structured_jsonl` message fixtures. It uses the first three corpus
seeds (7632647173703958409, 7968175640111700217, 912910298659544128), in fixture-major
order, with thinking disabled and the registered stochastic defaults unchanged.
Every request has a fixed 1024-completion-token cap. Each fixture first runs once
with the first seed and the same cap as an unmeasured graph warmup; all three
warmups precede the nine measured requests. Early EOS is retained.

```powershell
python3 tools/bench/compare_serve_windows.py --binary build/reference/ninfer-serve.exe --label reference
python3 tools/bench/compare_serve_windows.py --binary build/windows/apps/ninfer-serve.exe --label candidate --baseline build/serving-comparison/reference/report.json
python3 -m unittest tests.test_serve_comparison
```

Use Python 3.11 on Windows and an otherwise idle pair of GPUs. The helper starts
one hidden server on port 19080 with TP2 devices 0,1, MTP3 and the optimized draft
head, INT8 KV, 102400 context/capacity, chunk 1024, CUDA Graphs, vision enabled,
and concurrency one. It refuses an occupied port or an existing label directory
and stops only its own process on completion, failure, or interruption. The
default artifact comes from `config/model.json`; `--weights` selects an explicit
local path. No generated code is executed.

Reports live under `build/serving-comparison/<label>/`. `workload.json` freezes the
messages, order, caps, runtime options, and artifact/binary fingerprints before
launch. `records.jsonl` retains each warmup and measured response plus its matched
server event; `report.json` adds per-fixture means and sample standard deviations
and an optional baseline comparison. Console and structured server logs remain
alongside them; failed runs retain partial records and `failed.json`.

These are **short prompts with capped continuations**. The original prompts ask
for more work than fits in 1024 tokens: a `length` finish is retained, not treated
as task success or discarded. The helper applies no task-quality score. Exact
observable parity separately compares raw content, reasoning, tool functions and
arguments, finish reason, and completion count, excluding random response/tool
IDs. It also reports prompt-count, speculative-count, and prefix-count agreement.
Exit 1 means observable or prompt-count drift; exit 2 means an invalid run.

Shipping prefix reuse stays enabled, including history left by warmups. Reports
retain reused/computed prompt counts and reuse paths. PP tok/s uses only computed
tokens (no PP rate for a full cache hit); TG tok/s uses completion tokens minus
the first token divided by server decode time, following the existing corpus
convention. Server timings, MTP counters, and HTTP wall time are all retained.
Timing ratios remain separate from parity; changed output lengths, token
sequences, or prefix work change the workload and limit causal interpretation.
Three seeds in separate process runs supplement synthetic speed evidence; they
do not establish general throughput, 100K-context performance, or answer quality.

### Synthetic token workload

The benchmark slices exact token counts from `bench/fixtures/bench_corpus.ids`, calls
`Engine::prepare_tokens()`, then calls `Engine::generate()` once for each repetition. It does not
have a private prefill/decode loop and does not call target implementation interfaces.

The matrix contains three independently measured test kinds:

- `pp{P}` prepares `P` tokens and requests one output token. This is the smallest request that runs
  the model; `prefill t/s` is `P / GenerationTimings.prefill_seconds`.
- `tg{G}` prepares a one-token seed outside the reported phase and requests `G+1` output tokens.
  The begin-round token belongs to prefill, leaving exactly `G` tokens in the reported decode
  phase.
- `pp{P}+tg{G}` uses the same `G+1` convention after a `P`-token prefill and reports both phase
  rates from the same generation call.

All benchmark requests use raw output, disable model-default stops, and disable prefix reuse. This
keeps the requested token count exact without adding another generation path. When CUDA Graph is
enabled and the matrix contains decode work, one ordinary public generation request primes the
decode graph before warmups and measured repetitions.

## CLI

```text
ninfer_bench --weights <artifact.ninfer>
          [--corpus <ids-path>]
          [-p, --n-prompt <list>]
          [-n, --n-gen <list>]
          [-pg, --prompt-gen <P,G;P,G...>]
          [-r, --repetitions <n>] [--warmup <n>]
          [--max-ctx <tokens>] [--prefill-chunk <tokens>]
          [--kv-dtype <bf16|int8>]
          [--mtp-draft-tokens <0..5>] [--lm-head-draft]
          [--device <id>] [--tp <1|2>] [--devices <id,id>]
          [--no-cuda-graph] [--profile-measured]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.

Example:

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

`bf16` selects BF16 KV storage and `int8` selects INT8 group-64 KV storage. MTP is enabled with
`--mtp-draft-tokens`; `--lm-head-draft` selects the optimized proposal head. CUDA Graph decode is
enabled by default.

For two GPUs, pass `--tp 2 --devices 0,1`. Device order selects rank 0 first; an explicit
`--device` must agree with that first id. TP2 requires two distinct device ids and uses the same
public Engine path for prefill, decode, and MTP:

```bash
./build/bench/ninfer_bench \
  --weights /path/to/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --kv-dtype int8 \
  --mtp-draft-tokens 3 --lm-head-draft \
  -pg '2048,128' -r 3 --warmup 1 -o json --output-file tp2.json
```

JSON schema 12 records `config.tp`, ordered `config.devices`, and each GPU's name and ordinal
in `environment.devices`. CSV records `tp` and a quoted `devices` list. Timing definitions are
the same for both tensor-parallel degrees. Memory arena capacities and workspace peaks describe
rank 0; they are not total VRAM consumption across both GPUs.

`--profile-measured` is a benchmark-only profiler boundary. It requires exactly one selected test
and `-r 1`, synchronizes after warmup, and brackets only the measured repetition with
`cudaProfilerStart/Stop`. Use it with an Nsight Systems `cudaProfilerApi` capture range so artifact
load, graph construction, and warmup do not enter topology counts.

## Linear Op benchmark

`ninfer_linear_bench` measures only the public pure `linear()` contract. It supports Q4, Q5, Q6,
W8, registered BF16 weights, the registered NVFP4 problems, and the registered FP8 problems.
Existing formats use `--policy a16`; NVFP4 additionally supports `--policy a4`, and
FP8 supports `--policy a8`. Each permission lets the production resolver select the qualified
route for the exact geometry and T. LinearAdd, LinearSwiGLU,
LinearPair, Attention/GDN projections, and sparse MoE remain separate semantic Ops and are not
benchmark modes here.

This is a long-lived public benchmark: every timed and profiled point calls `ninfer::ops::linear`
and lets production dispatch choose the implementation. Candidate crossover work uses a
task-local temporary sweep, puts the winner or boundary in production dispatch, and deletes losing
candidates and temporary controls afterward; private launchers and route forcing do not belong in
this retained benchmark. An Op-scoped Linear qualification uses this benchmark directly and does
not load an artifact or invoke the target, Program, Engine, or round benchmarks documented
elsewhere in this file.

Build the benchmark and measure one exact production point:

```bash
cmake --build build --parallel --target ninfer_linear_bench
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
./build/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 14336 --k 5120 --t 1
./build/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 16384 --k 5120 --t 1024
```

A continuous small-T sweep reuses one packed weight and one maximum-T activation/output
allocation:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 \
  --sweep 1:32:1 --csv-out profiles/bench/q4_4096x5120_t1_32.csv
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 3456 --k 1152 \
  --sweep 4:512:4 --csv-out profiles/bench/q4_vision_qkv.csv
```

The registered suites run representative public Linear shapes for one or both exact products.
They are compact performance surveys, not copies of the production selector or numerical test
matrix:

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninfer_linear_bench --suite all
```

When a point group includes `T=1`, `T1_lin_x` reports the calculated ratio
`T * median(T=1) / median(T)`. It is a scaling reference from one public `T=1` benchmark point,
measured with the configured warmup and repetitions; the benchmark never issues T sequential
`T=1` launches as a substitute workload.

For NCU, `--profile` performs setup, warmup, and the L2 flush before enabling the profiler, then
captures exactly one public Linear call. That call may emit one or more production-selected kernel
launches:

```bash
ncu --profile-from-start off --set full \
  ./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8 --profile
```

Every ordinary sample is cold-cache: a 256 MiB L2 eviction write completes before the timed
interval. Reported effective bandwidth uses the encoded weight planes once, one BF16 activation
read, and one BF16 output write. Reported FLOPs are the mathematical `2*N*K*T`; neither metric
copies route-private tile, replay, padding, split, schedule, host-launcher, or kernel-instance
behavior. The fixed RTX 5090 memory reference is `1792 GB/s` DRAM bandwidth. Because `AllowA4` is
a permission rather than an execution-profile label, the long-lived benchmark does not infer or
report private activation compute or Tensor Core utilization. `READ_%` additionally compares the same one-read model
bytes with the measured `1674.5 GB/s` pure-read ceiling from `tools/hbm_bandwidth_probe.cu`; it is
the practical utilization measure for read-dominated points. Physical traffic and instruction
utilization still require NCU.

## Embedding Op benchmark

`ninfer_embedding_bench` measures the four registered quantized public `embedding()` profiles:
Q6 `[248320,5120]`, W8 `[248320,5120]`, W8 `[248320,2048]`, and row-scaled FP8
`[248320,5120]`. With no token override, each profile enumerates its exact aggregate Decode domain
for `B=1..8`: the three D=5120 profiles cover ordinary Decode and MTP through `T=48`, while
W8/D=2048 additionally covers DFlash through `T=128`.

Each interval contains one public Op call and receives one 256 MiB L2 eviction before timing; the
eviction itself is excluded. Effective bandwidth counts the selected encoded rows, their scales,
the I32 ids, and BF16 output once, and `READ_%` uses the measured RTX 5090 sustained-read reference.
There are no repeated-T=1 comparisons, private launchers, forced routes, candidate kernels, or
copied controls in this benchmark.

```bash
cmake --build build --parallel --target ninfer_embedding_bench
./build/bench/ninfer_embedding_bench --profile q6-d5120 --warmup 10 --repeat 61
./build/bench/ninfer_embedding_bench --profile w8-d5120 --warmup 10 --repeat 61
./build/bench/ninfer_embedding_bench --profile w8-d2048 --warmup 10 --repeat 61
./build/bench/ninfer_embedding_bench --profile fp8-d5120 --warmup 10 --repeat 61
./build/bench/ninfer_embedding_bench \
  --profile w8-d2048 --tokens 1,6,7,16,128 --warmup 10 --repeat 61 --csv
```

## GDN control-projection Op benchmark

`ninfer_gdn_gating_proj_bench` measures the registered BF16 control projection. With
`--norm-control`, it measures the complete 27B two-weight RMSNorm/control contract; adding `--35b`
selects the 35B contiguous-parent form. `--candidate auto` uses production dispatch;
`--candidate composed` is the explicit RMSNorm-plus-control comparison. Every row reports the
selected route and transient workspace after a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_gdn_gating_proj_bench
./build/bench/ninfer_gdn_gating_proj_bench \
  --norm-control --candidate auto \
  -p 1,2,3,4,5,6,8,16,32,48 --warmup 10 --repeat 200
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate auto \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
./build/bench/ninfer_gdn_gating_proj_bench \
  --35b --norm-control --candidate composed \
  -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --warmup 10 --repeat 200
```

## Gated DeltaNet Op benchmark

`ninfer_gated_delta_net_bench` measures the BF16 Gated DeltaNet contract with state/head dimension
128, production Q/K normalization, and any positive, divisible `value_heads >= qk_heads` mapping.
Running/chunked modes use batch 1; snapshot mode accepts exact `B=1..8` and optional mixed valid
prefixes. Every measurement is a CUDA Graph replay preceded by a 256 MiB L2 flush outside the timed
interval.

`--running` measures the public running-state entry across recurrent-only, complete 64-token
chunks, and chunked-plus-recurrent-tail routes. `--snapshot` measures the snapshot entry over the
production `W=1..16` batch range; `--qk-norm composed` retains the B=1 two-L2Norm comparison.
`--chunked-only` measures the complete pre-normalized BF16 pipeline through the public Op. Adding
`--breakdown` reports isolated `prepare_wy_wu`, `state_passing`, and `output` stage timings. These
three intrinsic algorithm stages are the benchmark's sole private-launcher exception; the complete
pipeline and every other mode remain public-contract calls.
`stage_share_pct` partitions the sum of isolated-stage medians. Each isolated stage receives its own
cold-L2 flush, so `relative_to_e2e_pct` is informative but is not an additive partition of the
pipeline latency.

`logical_bytes` and `logical_gbps` count each contract-visible tensor and state transfer once.
`traffic_bytes` and `traffic_gbps` instead sum one full tensor extent for every kernel input and
output in the selected implementation. This includes repeated consumption by different kernels,
the composed or public chunked Q/K-normalization intermediates, and every producer/consumer access
to chunked `g_cumsum`, W, U, `v_new`, and `h_chunk`. `intermediate_traffic_bytes` isolates those
normalization and chunked-workspace accesses. These deterministic byte counts describe
implementation-level tensor traffic; physical DRAM/L2 sectors and cache reuse still require NCU.

```bash
cmake --build build --parallel --target ninfer_gated_delta_net_bench ninfer_gdn_layer_bench
./build/bench/ninfer_gated_delta_net_bench \
  --running --value-heads 32 --sweep --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --snapshot --value-heads 32 --qk-norm fused --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --chunked-only --value-heads 32 --tokens 1024 --breakdown \
  --warmup 20 --repeat 100
./build/bench/ninfer_gdn_layer_bench \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --route fused --norm-control fused --qk-norm fused --warmup 20 --repeat 500
```

## GDN input-projection Op benchmark

`ninfer_gdn_input_proj_bench` measures all registered public `gdn_input_proj` forms: the 27B
Q4/Q5 two-parent projection, the 35B W8 single-parent projection, and the 27B NVFP4 or row-scaled
FP8 single-parent projection under its admitted policies. Every timed and profiled point is exactly
one public Op call. Single-parent workspace is queried and allocated through the public capacity
entry before timing; the benchmark has no private launchers, route controls, or candidate mode.

Cold cache is the primary model-layer condition. Reported logical traffic counts encoded weights
once, BF16 input once, and QKV/Z outputs once. FLOPs describe the complete registered projection.
The benchmark reports caller policy rather than inferring a private activation-compute route.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_bench
./build/bench/ninfer_gdn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 5 --repeat 30 \
  --csv-out profiles/bench/gdn_input_proj.csv
./build/bench/ninfer_gdn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 --cache cold --profile
./build/bench/ninfer_gdn_input_proj_bench \
  --format fp8 --fp8-policy a8 --tokens 1,2,3,4,5,6,7,8 --cache cold
```

## GDN input projection/convolution Snapshot/Record Op benchmark

`ninfer_gdn_input_proj_conv_snapshot_bench` measures the public Qwen3.6/Qwen3.8 Q4/Q5, NVFP4,
row-scaled FP8, and W8 `gdn_input_proj_conv_snapshot` / `gdn_input_proj_conv_record` forms for exact
`B=1..8`. The timed body is exactly one complete public Op call; the benchmark does not include
private launchers, candidate selection, duplicated compositions, or route labels. Its default
`T=1..6` sweep is the production MTP verification interval; Record begins at `T=2`.
`--form snapshot|record|both` selects the semantic form. NVFP4 accepts public `a16`/`a4`, while FP8
accepts `a16`/`a8`; the reported profile names caller policy, not a private resolved route.

CUDA Graph replay is the default execution mode. The graph contains external timing event nodes
around the complete Op body, while L2 eviction stays outside the timed interval. Cold-cache results
model successive model layers with distinct weights and are the authoritative comparison; warm
results make launch and cache effects visible. The initial state occupies a slot disjoint from all
published snapshot slots, so repeated replay does not introduce a benchmark-only state reset.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_conv_snapshot_bench
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format q4q5 --sweep 1:6 --execution graph --cache both \
  --warmup 10 --repeat 100 \
  --csv-out profiles/bench/gdn_input_proj_conv_snapshot.csv
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format nvfp4 --nvfp4-policy a4 --sweep 1:17 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format w8 --tokens 16 --batch 8 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format nvfp4 --tokens 6 --batch 3 --valid-columns 6,3,1 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format fp8 --fp8-policy a8 --form both --batch 1 --sweep 1:16 \
  --execution both --cache both --warmup 5 --repeat 30
```

`--execution eager|both` is available only to attribute launch behavior; it calls the same public
Op with the same operands and workspace.

## Softmax Attention Op benchmarks

The four Softmax Attention executables share one harness rule: fixture setup, public workspace
capacity queries, L2 conditioning, graph capture, and synchronization stay outside the timed
interval. An eager interval contains one public Op call; a captured graph contains that same one
public call. `--profile` likewise brackets one complete public call and requires one exact semantic
point. There are no private headers, launchers, route labels, candidate controls, tile sizes, split
counts, or kernel-name filters in these benchmarks.

`ninfer_causal_softmax_attention_bench` measures the two public causal-cache entries:
append-and-attend and cached-only. It covers the registered D256 H24/KV4 and H16/KV2 geometries
with BF16 and INT8-G64 KV storage. Production dispatch receives the caller-visible execution
envelope and owns all decode, prompt, Small-T, and split-KV choices.

Append-and-attend accepts `--batch 1,2,4,8`; each ordinary `--context L` point gives every row the
same context and all `W` columns are valid. One exact mixed profile uses `--row-contexts`,
`--valid-columns`, and `--table-rows`, each with exactly `B` entries. Cached-only remains B=1.
The timed call consumes the whole batch once; metadata copies and graph capture remain outside the
interval. Uniform full-width profiles use the dense public contract; exact partial profiles use
device-resident valid extents. Reported useful bytes/FLOPs sum only valid row work.

```bash
cmake --build build --parallel --target ninfer_causal_softmax_attention_bench
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry both --geometry all --kv-dtype all --batch 1 \
  --tokens 1,2,4,6,8,12,16 --context 0,128,2048,8192 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype int8 \
  --batch 3 --tokens 6 --row-contexts 127,2047,63 \
  --valid-columns 6,3,0 --table-rows 2,0,1 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 16 --context 8192 --execution graph --cache cold --profile
```

`ninfer_context_softmax_attention_bench` measures the public read-only context-plus-query contract
at Q32/KV8/D128 with BF16 context storage. `T` is a complete non-causal query block and `L` is its
external context length.

```bash
cmake --build build --parallel --target ninfer_context_softmax_attention_bench
./build/bench/ninfer_context_softmax_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,2048,8192,32768,131072,196608,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_sliding_window_attention_bench` measures the public Q32/KV8/D128 symmetric sliding-window
contract over the 4096-slot cyclic BF16 cache and a complete non-causal query block.

```bash
cmake --build build --parallel --target ninfer_sliding_window_attention_bench
./build/bench/ninfer_sliding_window_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,32,64,96,128,4095,4096,8192,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_packed_softmax_attention_bench` measures both public dense Attention overloads: a uniform
plain-segment entry and a packed entry driven by cumulative segment lengths. Equal-length inputs
can compare both public entries; nonuniform inputs select only `packed`.

```bash
cmake --build build --parallel --target ninfer_packed_softmax_attention_bench
./build/bench/ninfer_packed_softmax_attention_bench \
  --entry both --segments 16 --length 256 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_packed_softmax_attention_bench \
  --entry packed --segment-lengths 128,256,384,512 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

The bandwidth and FLOP fields are semantic useful-work models. They do not infer private scratch
traffic, launch decomposition, or selected implementation from kernel names; physical traffic and
instruction utilization require a profiler capture of the complete public call.

## KV cache append Op benchmark

`ninfer_kv_cache_append_bench` unifies the two public append contracts without combining them in
one timed body. `--mode full` calls full D256 KV publication for KV4/KV2 and BF16/INT8-G64 caches.
`--mode prefix` calls device-count prefix publication for BF16 D128/KV8 linear or 4096-slot cyclic
caches; `T` is the public envelope and `C` is the device commit count. Every measured interval or
captured graph contains exactly one selected public append call.

```bash
cmake --build build --parallel --target ninfer_kv_cache_append_bench
./build/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry all --kv-dtype all --tokens 1,2,4,8,16 \
  --context 128 --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_kv_cache_append_bench \
  --mode prefix --tokens 1,2,4,8,16 --counts 0,1,2,4,8,16 \
  --layout all --execution graph --cache cold --warmup 10 --repeat 61
```

Prefix useful traffic is 8192 bytes per committed token; `C=0` still exercises the public
device-count contract and reports zero useful bytes.

## Masked-block preparation Op benchmark

`ninfer_prepare_masked_block_bench` measures the public exact I32 anchor/mask transform for every
registered `B=2..16`. Eager and graph modes call the same public contract; cold mode performs the
256 MiB L2 eviction before the timed interval.

```bash
cmake --build build --parallel --target ninfer_prepare_masked_block_bench
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --execution graph --cache both --warmup 20 --repeat 101
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 16 --execution graph --cache cold --profile
```

Useful traffic is `(2+2B)*4` bytes: two device scalar reads and two complete I32 output writes.

## W8 LinearSwiGLU Op benchmark

`ninfer_w8_linear_swiglu_bench` measures the registered W8 `[12288,2048] -> [6144,T]`
LinearSwiGLU profile. Production writes only the fused output and uses no workspace. The explicit
control runs the registered parent `linear` followed by `silu_mul`. Candidate mode retains the
decode, exact-T split-K Tensor Core, and tiled Tensor Core schedules used to tune every dispatch
seam. Every cold-cache sample follows a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_w8_linear_swiglu_bench
./build/bench/ninfer_w8_linear_swiglu_bench \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,96,128,256,512,896,1024 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_linear_swiglu.csv
./build/bench/ninfer_w8_linear_swiglu_bench \
  --profile --t-sweep 1024
```

## Q4 LinearSwiGLU Op benchmark

`ninfer_q4_linear_swiglu_bench` measures the public Q4 `[34816,5120] -> [17408,T]` profile and
queries workspace capacity for the requested aggregate interval.

```bash
cmake --build build --parallel --target ninfer_q4_linear_swiglu_bench
./build/bench/ninfer_q4_linear_swiglu_bench \
  --t-sweep 1,2,4,8,16,24,32,40,48 --warmup 10 --repeat 50
```

## FP8 LinearSwiGLU Op benchmark

`ninfer_fp8_linear_swiglu_bench` measures the public row-scaled FP8 `[34816,5120] ->
[17408,T]` profile. `--policy a8` measures the production resolver, including caller-owned
activation workspace and the fused SwiGLU output; `--policy a16` measures the public A16 form.
The Tensor Core percentage uses the RTX 5090 dense FP8/FP32-accumulate reference of 419 TFLOP/s
only for extents that the production resolver sends to A8.

```bash
cmake --build build --parallel --target ninfer_fp8_linear_swiglu_bench
./build/bench/ninfer_fp8_linear_swiglu_bench \
  --policy a8 \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,40,48,1024 \
  --warmup 5 --repeat 30
```

## Q5 LinearAdd Op benchmark

`ninfer_q5_linear_add_bench` measures the public Q5 `[5120,6144]` and `[5120,17408]`
LinearAdd profiles. Every cold-cache sample is one production-dispatched public call, with the
workspace capacity queried for the requested interval.

```bash
cmake --build build --parallel --target ninfer_q5_linear_add_bench
./build/bench/ninfer_q5_linear_add_bench \
  --k 6144 --t-sweep 1,2,4,8,16,24,32,48 --warmup 10 --repeat 50
./build/bench/ninfer_q5_linear_add_bench \
  --k 17408 --t-sweep 1,2,4,8,16,24,32,48 --warmup 10 --repeat 50
```

## BF16 LinearAdd Op benchmark

`ninfer_bf16_linear_add_bench` measures the contiguous BF16 `[5120,6144]` projection with its
in-place BF16 residual epilogue. Production uses decode at `T=1`, exact-small-T at `T=2..4`,
aggregate MMA through `T=48`, and the large-T MMA afterward.
Every sample is cold-cache. Effective bandwidth counts the weight once, the activation once, and
the residual read plus write; its `READ_%` and `TC_%` use the benchmark's explicit RTX 5090 BF16
references.

```bash
cmake --build build --parallel --target ninfer_bf16_linear_add_bench
./build/bench/ninfer_bf16_linear_add_bench \
  --sweep 1:48:1 --route production --warmup 10 --repeat 50 \
  --csv-out profiles/bench/bf16_linear_add_t1_48.csv
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024,1536,2048 --route production --warmup 10 --repeat 50
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024 --route production --profile
```

## W8 LinearAdd Op benchmark

`ninfer_w8_linear_add_bench` measures the W8 `[2048,4096]` and `[2048,6144]` projections with their
BF16 residual epilogue. Production updates the residual in place and uses no workspace. Use
`--production-only` for public evidence; every cold-cache sample follows a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_w8_linear_add_bench
./build/bench/ninfer_w8_linear_add_bench \
  --k 4096 --production-only \
  --t-sweep 1,2,4,8,16,32,48,64,96,128 --warmup 10 --repeat 50
./build/bench/ninfer_w8_linear_add_bench \
  --k 6144 --production-only \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,64,65,96,128,256,640,641,896,960,1024,1280,2048 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_linear_add.csv
./build/bench/ninfer_w8_linear_add_bench \
  --production-only --t-sweep "$(seq -s, 1 2048)" \
  --warmup 3 --repeat 15 --csv-out profiles/bench/w8_linear_add_all_t.csv
./build/bench/ninfer_w8_linear_add_bench \
  --profile --production-only --t-sweep 1024
```

## FP8 LinearAdd Op benchmark

`ninfer_fp8_linear_add_bench` measures the two public row-scaled FP8 LinearAdd registrations,
including activation quantization, caller-owned workspace, contraction, residual read, and final
in-place BF16 write. `--policy a8` follows the independent production resolver of the selected
semantic Op: `[5120,6144]` uses A16 below `T=22`, while `[5120,17408]` uses A16 below `T=25`; larger
extents use FP8/FP32-accumulate Tensor Core contraction. `TC_%` is reported only when that A8 route
actually executes, against the RTX 5090 419 TFLOP/s reference.

```bash
cmake --build build --parallel --target ninfer_fp8_linear_add_bench
./build/bench/ninfer_fp8_linear_add_bench \
  --k 6144 --policy a8 --t-sweep 1,2,4,8,16,20,21,22,32,48,1024 \
  --warmup 5 --repeat 30
./build/bench/ninfer_fp8_linear_add_bench \
  --k 17408 --policy a8 --t-sweep 1,2,4,8,16,24,25,32,48,1024 \
  --warmup 5 --repeat 30
```

## Attention input-projection Op benchmark

`ninfer_attn_input_proj_bench` measures every registered public `attn_input_proj()` weight/shape
contract: the 27B two-parent Q4/Q5 projection; the 35B W8 Q/K/gate/V and companion Q/K/V
projections; and the 27B BF16, NVFP4, and T=1 FP8 single-parent Q/K/gate/V projections. Fixture
packing and public workspace capacity queries happen before timing. Every sample and profiler
range contains exactly one public Op call, so production owns format-specific dispatch and launch
decomposition.

```bash
cmake --build build --parallel --target ninfer_attn_input_proj_bench
./build/bench/ninfer_attn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 10 --repeat 50 \
  --csv-out profiles/bench/attn_input_proj.csv
./build/bench/ninfer_attn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 \
  --cache cold --warmup 10 --profile
./build/bench/ninfer_attn_input_proj_bench \
  --format fp8 --fp8-policy a8 --tokens 1 \
  --cache cold --warmup 10 --repeat 50
```

The stateful GDN projection/convolution/snapshot contract remains in its own public Op benchmark;
it is not a mode of Attention input projection. End-to-end target measurement uses `ninfer_bench`.

## 35B sparse-MoE dFlash benchmark

`ninfer_sparse_moe_bench` measures the complete routed-plus-shared post-mixer Op for the 35B Text
Q4+Q5/Q6 profiles and the MTP W8+W8 profile. It is a long-lived public Op benchmark: fixture setup
uses `sparse_moe_workspace_capacity_bytes()`, and every eager or captured measurement calls only
`ninfer::ops::sparse_moe()`. Production dispatch exclusively owns decode, Small-T, prefill,
workspace views, launch decomposition, and schedule selection.

CUDA Graph replay is the default and authoritative execution mode. The benchmark eagerly
materializes the public call, captures it with stable tensor and workspace addresses, instantiates
the graph, and primes one replay before configured warmup. Timing-enabled external event nodes
surround the captured public call, so `graph_replay` reports the complete device-side SparseMoe
body while excluding fixture reset, L2 eviction, graph capture/instantiation/prime, host launch,
and host synchronization. `eager` uses the same public-call lambda and is only a comparison mode.

```bash
cmake --build build --parallel --target ninfer_sparse_moe_bench
./build/bench/ninfer_sparse_moe_bench \
  --codec q4-q5 --tokens 1 --execution graph --cache both \
  --distribution trace-like --warmup 20 --repeat 200
./build/bench/ninfer_sparse_moe_bench \
  --codec q4-q5 --sweep 1:128:1 --execution graph --cache cold \
  --distribution trace-like --warmup 5 --repeat 50 \
  --csv-out profiles/bench/sparse_moe_public_graph.csv
```

Each cold sample flushes 256 MiB before the timed interval. `trace-like` is the primary
single-sequence verification distribution; `independent` and `same` bound zero and complete expert
overlap. `--execution eager|graph|both` and `--cache cold|warm|both` change only the harness around
the same public call. The CSV `timed_scope` is `full_sparse_moe_device_body`; it contains no private
route, candidate, grid, block, or launch-count fields.

`unique_weight_gbps` counts the router, shared weights, and each selected expert's encoded weights
once, so it is a distribution-aware useful-traffic lower bound rather than measured DRAM traffic.
Its peak percentage uses the device's theoretical DDR bandwidth. `logical_tflops` counts the
router plus eight routed and one shared gate/up and down matrix products.

## DFlash LinearPair benchmark

`ninfer_linear_pair_bench` measures one public `linear_pair()` call over the exact adjacent W8
`[1024,2048]` K/V row views. It uses cold-cache CUDA Graph replay, accepts an arbitrary token list
or sweep, and reports route-neutral effective bandwidth, logical FLOP/s, and calculated T=1 linear
extrapolation.

```bash
cmake --build build --parallel --target ninfer_linear_pair_bench
./build/bench/ninfer_linear_pair_bench --sweep 1:128:1 --warmup 5 --repeat 30
```

## MTP exact-transform benchmark

`ninfer_mtp_pack_bench` calls the public bit-exact pack or attention-split Op once per point:

```bash
cmake --build build --parallel --target ninfer_mtp_pack_bench
./build/bench/ninfer_mtp_pack_bench --op pack --d 5120 --tokens 1,2,3,4,5,6,48
./build/bench/ninfer_mtp_pack_bench --op split --tokens 1,2,3,4,5,6,48
```

## Target MTP round benchmark

`ninfer_qwen3_6_27b_mtp_round_bench` measures the registered target's native proposal and
verification round without introducing a second generation controller. It loads the same `.ninfer`
artifact through the target-private package facade, prepares a real prompt with that target's
Frontend, and reports draft/accept statistics for the target-owned MTP schedule:

```bash
cmake --build build --parallel --target ninfer_qwen3_6_27b_mtp_round_bench
./build/bench/ninfer_qwen3_6_27b_mtp_round_bench \
  --artifact out/qwen3_6_27b.ninfer
```

## 35B complete DFlash round benchmark

`ninfer_qwen3_6_35b_a3b_dflash_round_bench` drives the production Program through consecutive
steady DFlash rounds. A measured round includes the previous confirmed feature-to-context append,
the six-layer proposal, target verify/accept, and host publication. It reports GPU and wall latency,
real acceptance, per-position acceptance, mean licensed length, and published tokens/s:

```bash
cmake --build build --parallel \
  --target ninfer_qwen3_6_35b_a3b_dflash_round_bench
./build/bench/ninfer_qwen3_6_35b_a3b_dflash_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 4096 --draft-tokens 15 --proposal-head optimized
```

Run separate invocations for `K=1..15`, `full|optimized`, representative contexts, and
`--no-cuda-graph`. The target-side benchmark above supplies forced `A=0..K` transaction evidence;
this complete-round benchmark intentionally preserves the DFlash model's real greedy acceptance.

## Token-decision Op benchmarks

The G1 benchmark calls public `argmax` for the Qwen3.6-35B full physical vocabulary with 248077
valid rows through `C=128`, and for the 131072-row shortlist through `C=120`. With no arguments it
covers every B=1 full-vocabulary width, both T1 routes, and both aggregate maxima:

```bash
cmake --build build --parallel --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/bench/ninfer_argmax_bench
./build/bench/ninfer_argmax_bench --shape full --cols 128
./build/bench/ninfer_argmax_bench --shape shortlist --cols 120
```

The G2/G3 benchmark uses physical rows 248320, valid token domain 248077, optional occurrence
counts, batched sampling at `B=1,2,4,8`, and every MTP window `K=1..5`. With no arguments it runs
the full greedy/stochastic matrix; individual routes are suitable for Nsight Compute capture:

```bash
./build/bench/ninfer_sampling_select_bench --matrix
./build/bench/ninfer_sampling_select_bench --sample --batch 8 --mode stochastic --top-k 20
./build/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
```

## 35B dFlash causal Attention qualification

The public causal benchmark covers exact verify widths `W=1..16`, both KV codecs, and the
append-and-attend and already-cached entries for the D256 H16/KV2 geometry. Batched target
qualification uses the append entry:

```bash
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype bf16 \
  --batch 1,2,4,8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
```

The former `attention_layer` executable composed multiple implementation-level stages and exposed
private route controls, so it is not retained as a public Op benchmark. Complete mixer and target
effects are measured through the public Engine benchmark or the target round benchmarks.

## Local TP2 attention split experiment

`ninfer_gqa_tp2_split_bench` isolates the production INT8 partial/reduce kernels for a 12-query,
2-KV-head TP2 shard (head dimension 256). It compares fixed split caps with the production 70-SM
broad-graph policy at T=1, 4 and 5. The `sm70_broad_graph` row launches the original 170 splits per
KV head and selects 70 active splits from device positions in the qualified 81,920..131,077
visible-key interval. Outside that interval the production policy keeps its original split count.

```bash
cmake --build build --parallel --target ninfer_gqa_tp2_split_bench ninfer_gqa_tp2_sm70_test
./build/bench/ninfer_gqa_tp2_split_bench --device 0 --context 100000 --repeat 31
./build/bench/ninfer_gqa_tp2_split_bench --device 1 --context 100000 --repeat 31
./build/tests/ninfer_gqa_tp2_sm70_test 0
./build/tests/ninfer_gqa_tp2_sm70_test 1
```

Configure with `NINFER_BUILD_BENCHMARKS=ON` and `BUILD_TESTING=ON`. Native Windows uses the
corresponding `.exe` paths under the selected build directory. The experiment links only core
helpers; the public-Op regression links the production launchers.

Every candidate is checked against the existing independent FP64 oracle using represented BF16
queries and decoded INT8-G64 cache, across all 12 heads and query tokens. It then times cold-cache
CUDA Graph replay with interleaved candidates, ten warmup replays, and a 128 MiB flush outside the
CUDA event interval. CSV reports minimum, median, p90, relative L2 error and its criterion. The
production compiler's relocatable-device-code mode is enabled for this benchmark.

The tuning applies to one active request on a 70-SM `sm_120` GPU, INT8 cache and T=1/4/5. The local
100K measurements reduce attention latency by 13-22%; this isolated Op result does not measure
TP communication, prefill throughput, model accuracy or an end-to-end inference speedup. See
[the local measurements](../docs/performance.md#local-rtx-5070-ti-attention-tuning).

## Pointwise Op benchmarks

The Section 5 benchmarks cover the complete Qwen3.6-35B pointwise matrix. Default invocation runs
all registered small, established, maximum-video, and maximum-image shapes. `--control` preserves
the selected kernel topology and payload while replacing the mathematical operation with minimal
bitwise work:

```bash
cmake --build build --parallel --target \
  ninfer_residual_add_bench ninfer_sigmoid_mul_bench \
  ninfer_gelu_bench ninfer_add_bias_bench

./build/bench/ninfer_residual_add_bench [--patches P] [--control]
./build/bench/ninfer_sigmoid_mul_bench \
  [--tokens T[,T...]] [--control | --candidate-block B]
./build/bench/ninfer_position_bench \
  [--tokens T[,T...]] [--candidate-block B] [--cold-graph] [--warmup N] [--repeat N]
./build/bench/ninfer_gelu_bench [--mode tanh|exact --columns C] [--control]
./build/bench/ninfer_add_bias_bench [--d D --columns C] [--control]
```

Aligned registered shapes use 16-byte BF16 packs in the cache-sized regime. GELU and AddBias
select their BF16x2 streaming routes for larger Vision items; odd or unaligned repository-internal
test shapes exercise the scalar fallbacks.

## Reports

Table, JSON, and CSV reports all identify the selected target, artifact, Engine configuration,
load summary, memory capacity, KV payload, workspace peak, phase throughput, and speculative
statistics. JSON schema version 10 records the public value objects directly:

- `load`: target, `weights_id`, load/upload time, file/H2D/staging bytes, tensor count, and resource
  count;
- `memory`: weights/sequence/workspace/request-transient arenas, planned context, KV storage,
  CUDA Graph allowance, and KV payload;
- each repetition's `timings`: prepare, Vision, prefill, decode, and total seconds;
- each repetition's `speculative`: window, rounds, drafted/accepted tokens, fallbacks, and per-position
acceptance.

Each test reports `workspace_peak_bytes` from the planned phase markers, including CUDA Graph
replay, and `workspace_allocator_peak_bytes` from host-side arena allocation activity. These are
intentionally separate: replay reuses captured addresses without advancing the host allocator.

`decode_output_tok_s` counts the requested `G` decode outputs. `decode_engine_tok_s` uses the
Program's speculative round statistics, so it also describes work performed by a final partially
committed speculative round. Reports also contain the command and machine information needed to
interpret a local measurement.

Raw reports and profiler captures remain local under `profiles/bench`, `profiles/ncu`, and
`profiles/nsys`.

### Eager TP2 transfer comparison

`ninfer_peer_transfer_bench [collectives-per-batch pairs]` compares CUDA-managed
cross-device copies with explicit portable pinned staging on a pair without P2P.
Defaults are 128 consecutive reductions per batch and five alternating-order pairs,
after eight warmup reductions per route. Payloads are 64 KiB, 1 MiB, and the real
10 MiB `[5120,1024]` BF16 prefill shape; allocation and input reset are outside timing.

Each JSON sample records host enqueue time, complete wall time, per-rank CUDA event
time, and a device-0 event interval joined after device 1 completes. Cross-device
events establish the dependency; elapsed-time queries always use events on one GPU.
These quantities overlap and must not be summed. This is an eager collective benchmark,
not model throughput or a promise that an Async API returns immediately.

Run `ninfer_peer_transfer_test` first. It separately checks nonzero changing operands,
the FP64 sum/storage oracle, asymmetric exact row gathers, host-buffer reuse, guards,
independent owners, reversed ranks, and captured fallback. Accept a transport change
only after the paired timing and unprofiled model prefill/quality checks pass.

### TP2 target-logit column gather experiment

`ninfer_allgather_columns_bench` compares the per-column baseline with one packed
peer transfer plus an exact integer interleave. `--with-direct2d` adds an experimental
pair of pitched CUDA copies per rank. Both ranks retain the full physical vocabulary,
including padding; the benchmark performs no logit arithmetic, sampling or filtering.
The target-logit runtime selects the qualified packed Op, with T1 retaining its existing
gather path. Model response parity, prefix checks and end-to-end measurements passed;
see the [runtime record](../diagnostics/column-gather-runtime-validation.json).

```powershell
cmake --build build/windows -j --target ninfer_allgather_columns_test ninfer_allgather_columns_bench
build/windows/tests/ninfer_allgather_columns_test.exe
build/windows/tests/ninfer_allgather_columns_test.exe --reverse-devices
build/windows/tests/ninfer_allgather_columns_test.exe --candidate-direct2d
build/windows/tests/ninfer_allgather_columns_test.exe --candidate-direct2d --reverse-devices
build/windows/bench/ninfer_allgather_columns_bench.exe --cols 4 --with-direct2d
build/windows/bench/ninfer_allgather_columns_bench.exe --cols 1 --with-direct2d
```

Linux uses the corresponding executable paths without `.exe`. The test's default route
is the public packed Op; `--baseline` checks the existing per-column composition. It
compares raw BF16 bits against independent host concatenation, including NaN payloads,
signed zeros, asymmetric shard widths, input/guard preservation, changing consecutive
calls, both eager transports, captured replay, and whole-graph executable updates between
distinct live bindings of the same shape. T1 and T4 are separate graph topologies. An
unsupported direct-2D capture or update reports a failure; it never silently substitutes
the packed candidate.

The benchmark fixes the actual shard width at 124160 and supports T1 or T4, with resident
raw BF16 inputs. Default timing is ten warmups and 31 rotating paired samples of one
gather, in both eager and captured modes. `--mode eager|graph|both`, `--calls N`,
`--samples N`, `--warmup N`, `--reverse-devices`, and `--implicit-only` bound individual
comparisons. Portable pinned staging is provisioned by default and remains eager-only;
captured calls use the existing UVA copy path. No mailbox is installed or reset.
Every selected route passes its own poisoned-output exact oracle before and after
measurement; final timed outputs and guards are checked before any reissue.
JSON reports host enqueue, complete wall and joined-device time per gather, including
median/p95 summaries. These overlapping intervals are not additive. This is gather
latency; committed-token throughput is measured separately in the model benchmark.

The packed Op's explicit per-rank scratch query is
`allgather_columns_workspace_capacity_bytes(peer_rows, T)`: zero for T1 and 993280 bytes
for the 124160-row T4 peer shard. `TextContext::logits_tp2` uses this Op; `tp_call_roots`
and the full-head MTP fallback in `tp_mtp_call_roots` explicitly plan its aligned scratch
alongside the live vocabulary half. The same-arena regression checks nonzero allocation
cursors, input and guard preservation, exact output, and the planned versus actual peak.
The optimized distributed draft argmax and persistent full-logit outputs are unchanged.

On this Windows dual 5070 Ti host, the packed candidate passed both rank orders and
Compute Sanitizer with zero errors. At T4, median captured complete-wall latency
was 821.6 us for the per-column baseline and 401.5 us for packed (31 rotating paired
samples). T1 retains the same gather path. The direct 2D experiment passes eager T4
but fails whole-graph update with cudaErrorGraphExecUpdateFailure (result 5), so it
is not qualified for production here. [Operator evidence](../diagnostics/column-gather-op-validation.json).
The separate runtime comparison measured 2.83% and 2.58% generation gains at 8K and
100K on the fixed synthetic corpus, with unchanged speculative counts and exact
18-case response / four-case retrieval parity. Overall planned workspace and the
reported allocator peak stayed at 202304000 and 117743616 bytes; the 993280-byte
peer shard is explicitly planned even though another stage determines the peak.
[End-to-end method and results](../docs/performance.md#local-dual-5070-ti-packed-target-logit-gather).

### TP2 T4 NVFP4 fused SwiGLU CTA grouping

`ninfer_nvfp4_swiglu_tp2_cta_bench --device 0` (then `--device 1`) measures
only the [17408,5120] TP2 gate/up shard with four BF16 activation columns. It
compares the normal production private shard launcher (8 warps/CTA) with separate
benchmark instantiations at 8, 4 and 16 warps. The single-device public operator
does not admit this shard shape. Production dispatch and shape admission remain
unchanged; sharing the unchanged private kernel body is the only production refactor.

Both fixed seeds use dense, mixed-sign, O(1)-RMS represented BF16 activations on
all four tokens and coordinate-decorrelated NVFP4 weights. Every output is checked
against the independent represented-weight FP64 dot/SiLU/product oracle with the
existing A16 criterion, and every candidate must match production BF16 bits exactly.
Eager calls and captured replays are qualified; guards and input bytes are checked
before/after timing. `--qualify-only` performs these checks without the timing loops.

Default timing uses ten warmup rounds and 31 paired sample rounds, rotating route
order and reversing each four-round cycle. Each graph contains one measured kernel
between external timing-event nodes. A separate condition reads/writes 128 MiB
before the timed interval to perturb caches; both warm and scrubbed results are
reported. Allocation, oracle computation and graph capture are excluded. JSON
lines contain every paired sample and median/p95 summaries. The event interval
excludes the cache scrub; complete graph wall time includes it and host overhead.
Neither cache condition establishes model throughput or physical memory bandwidth.

`--samples N` and `--warmup N` change measurement counts. Run each physical GPU
separately; do not overlap with inference. A candidate needs existing normal/tail
shape regressions plus model response parity and unprofiled end-to-end gains before
changing the production default. Library-versus-benchmark 8-warp parity checks
post-extraction code generation; it is not a pre-extraction binary comparison.

Both local GPUs retained the eight-warp winner. Across the two fixed seeds, warm
and cache-scrubbed paired kernel times made four warps about 14-17% slower and
sixteen about 30-37% slower. All four routes passed the FP64 and exact-bit checks.
No production schedule change or inference gain follows from this experiment.
[Measurement record](../diagnostics/swiglu-cta-validation.json).


### TP2 T4 FP8 GDN activation access

`ninfer_fp8_gdn_tp2_access_bench --device 0` (then `--device 1`) compares
the actual production shard launcher with private `SharedPhase` and
`TokenPacked` instantiations for [8192,5120], T4. All routes retain eight warps,
two rows per warp, sixteen values per lane, one accumulator chain and the same
Q/K/V/Z output policy. Production kernels and dispatch remain unchanged.

Both fixed seeds use dense represented BF16 values in all four columns.
Every output passes an independent decoded-weight FP64 linear oracle, checked
separately for Q, K, V and Z with the existing FP8 A16 criterion, and must match
production BF16 bits exactly. Guards and immutable inputs are checked in eager
execution, captured replay and resident final timed outputs before reissue.

Timing uses ten warmup rounds and 31 rotating three-way paired rounds, with both
warm and 128 MiB cache-scrubbed conditions. External CUDA events bracket only the
kernel; complete graph wall time includes the scrub and host overhead.
`--samples N`, `--warmup N` and `--qualify-only` have the same roles as in the
SwiGLU experiment above. Run each physical GPU separately without other GPU work.

Both GPUs passed all qualifications. TokenPacked's median paired kernel latency
ratio to production SharedPhase was:

| GPU | Seed | Warm | 128 MiB scrubbed |
| --- | --- | --- | --- |
| 0 | 1803 | 0.8898 | 1.0069 |
| 0 | 1811 | 0.8810 | 1.0327 |
| 1 | 1803 | 0.9219 | 1.0078 |
| 1 | 1811 | 0.9150 | 1.0039 |

A ratio below one means lower latency: TokenPacked was 7.8-11.9% faster warm
and 0.4-3.3% slower after scrubbing. The complete-engine A/B at 8192 prompt /
256 generated tokens measured 205.23 +/- 0.14 versus 205.39 +/- 0.13 tok/s
(mean +/- sample standard deviation, three repetitions after one warmup). The
0.08% difference does not establish a gain. All 18 response observables and MTP
counts matched. The temporary runtime candidate was removed; SharedPhase remains
the production schedule. [Measurement record](../diagnostics/fp8-gdn-access-validation.json).

The GDN input-projection numerical test passed on both GPUs, including FP8 T4,
and `ninfer_gdn_projections_split_test --fused-input-only` passed the affected
TP2 fused projections. The default full split suite still reproduces an inherited
TP1 BF16 gating T1024 failure: its 192-CTA cooperative launch exceeds the local
shared-memory upper bound of 140 simultaneous CTAs. The explicit focused mode does not turn that
default-suite failure into a pass.

### Rank-zero speculative acceptance experiment

`ninfer_speculative_rank0_bench` compares the complete existing packed allgather, argmax, and
acceptance on both ranks with a one-destination gather, rank-zero argmax/unchanged acceptance,
and exact decision replication. It uses the real padded vocabulary 248320 (248077 logical),
K=3, B=1, extent=3, both device streams joined, and 31 rotating paired samples by default.
Greedy and nondegenerate stochastic configurations run separately; the latter retains the
original sampler, penalties, RNG key, and occurrence-counter effects. This is an isolated
schedule experiment, not an end-to-end engine throughput claim.

```powershell
.\build\windows\bench\ninfer_speculative_rank0_bench.exe --sampling both --mode both
.\build\windows\bench\ninfer_speculative_rank0_bench.exe --sampling both --mode graph --no-mailbox
```

The default eager gather uses explicit pinned staging; graphs use the captured copy path and,
when enabled, a compact one-way mapped mailbox for the decision. The benchmark checks actual
mailbox flags after both GPUs retire and reports enqueue, joined-device, and complete wall time.
Logit shards stay resident. Frontier/counter reset, output poisoning, and mailbox reset occur
outside each measured interval for both routes. Independent raw-bit gather/argmax checks and
state/counter invariants run before timing, on each route's final timed state before reset, and
after a fresh poisoned replay; complete decisions must match the baseline exactly. Probability
qualification remains the existing FP64 sampler/speculative-round oracle suite.

`ninfer_speculative_rank0_test [--reverse-devices]` adds changed no-sync calls, shortened extents
including zero, duplicate licensed tokens, mixed greedy/stochastic rows, null counters, asymmetric
shards, exact guards, and replay/whole-graph update between two live owners. The test
passed in both device orders; Compute Sanitizer memcheck with stream-ordered race tracking
reported zero errors. Existing exact-gather and independent FP64 speculative tests also passed.

The K3/B1 decision transfers 32 bytes: four licensed tokens, frontier, anchor,
licensed count and accepted count. Each rank's decision arena bound is 256 bytes;
the one-way T4 logit gather needs 993280 bytes of rank-zero scratch and none on
rank one. These are primitive bounds, not a measured reduction in total engine memory.

Measured median complete-pipeline wall time on this Windows dual 5070 Ti host:

| Execution / decision transport | Sampling | Both ranks (us) | Rank zero (us) |
| --- | --- | ---: | ---: |
| Eager / event copy | greedy | 321.9 | 391.2 |
| Graph / mailbox | greedy | 394.5 | 265.7 |
| Eager / event copy | stochastic | 318.2 | 395.4 |
| Graph / mailbox | stochastic | 439.3 | 306.6 |
| Graph / event-copy fallback | greedy | 409.7 | 307.2 |
| Graph / event-copy fallback | stochastic | 445.4 | 358.4 |
| Graph / mailbox, reversed devices | greedy | 403.3 | 286.0 |
| Graph / mailbox, reversed devices | stochastic | 437.9 | 313.6 |

Captured execution improves with the mailbox, its event-copy fallback and reversed
device order. Eager execution is slower. The qualified runtime now selects
rank-zero acceptance during CUDA capture and retains replicated acceptance eagerly.
These resident-logit timings remain acceptance-pipeline measurements; the separate
[full-engine qualification](../diagnostics/rank0-acceptance-runtime-validation.json)
records response parity and 8K/100K throughput.
[Operator validation and timing record](../diagnostics/rank0-acceptance-op-validation.json).

## Local TP2 mailbox geometry experiment

`ninfer_mailbox_geometry_bench --devices 0 1 --qualify-only` checks the actual
private mailbox sum kernel in place, including every output from 137 exchanges,
changing inputs, both producer skew directions, independent FP64/BF16 arithmetic,
published payloads, counters, flags and guards. Run the default 31 paired samples
with `--devices 0 1`, then `--devices 1 0`. Retired timeout flags are checked after
every replay; reset/input restore are excluded from timing. The captured end event
joins both ranks. This target matches production RDC compilation.

The 40 KiB experiment compares production `3x256` against `1x256`, keeping grouping,
fences, mailbox ownership and per-element addition unchanged. The 10 KiB control
uses the same one-block launch on both routes. Local paired median joined latency
was 1.6-2.3% higher with the 40 KiB one-block candidate across both device orders
and skew settings, so **production remains at three blocks**. No inference speed
claim or runtime change follows from this homogeneous exchange-chain benchmark.
See [qualification and timings](../diagnostics/mailbox-geometry-validation.json).


### TP2 prompt down-projection TMA token-tile experiment

`ninfer_nvfp4_down_tp2_tma_bench` is a private schedule experiment for the NVFP4
TP2 down projection `[N=5120,K=8704,T=1024]`. It compares the existing
`M256,N128,K128,S3,min1` schedule against `M128,N128,K128,S3,min1`,
`M128,N128,K128,S2,min1` and `M128,N128,K128,S2,min2`; production dispatch is unchanged. The token-tile
change increases the grid from 160 to 320 CTAs and changes shared/register
requirements. These are candidate tradeoffs, not a demonstrated performance gain.

```powershell
cmake --build build/windows --target ninfer_nvfp4_down_tp2_tma_bench -j
.\build\windows\bench\ninfer_nvfp4_down_tp2_tma_bench.exe --device 0 --qualify-only
.\build\windows\bench\ninfer_nvfp4_down_tp2_tma_bench.exe --device 1 --qualify-only
.\build\windows\bench\ninfer_nvfp4_down_tp2_tma_bench.exe --device 0 --samples 31 --warmup 10
.\build\windows\bench\ninfer_nvfp4_down_tp2_tma_bench.exe --device 1 --samples 31 --warmup 10
```

All routes instantiate the existing TMA kernel, preserving the activation
quantizer, K128 traversal, the two K64 MMA steps, FP32 accumulator, scale factor,
and output conversion. The identity epilogue models the rank-one partial; the
residual epilogue models rank zero's in-place `BF16(FP32_accumulator * alpha +
FP32(BF16_residual))` before the unchanged TP reduction. Changing independent
output grouping is expected to preserve the result, but exact identity is a test
requirement rather than an assumed guarantee. This executable uses **non-RDC**
compilation, matching `ninfer_nvfp4_tma`: compiling the M256 warp-specialized
kernel under RDC discards its `setmaxnreg` register-allocation contract.

For each of the two fixed seeds (1803 and 1811), every public BF16 activation is
dense and mixed-sign. The public `linear` and `linear_add` AllowA4 outputs are
qualified first against an independent FP64 oracle. The oracle directly decodes
the signed packed weight and its stored scale and sums all 8704 products from the
represented BF16 activation; residual is added from its represented BF16 value.
It does not reproduce private A4 input quantization or intermediate accumulation
rounding. The existing A4 criterion is unchanged: relative L2 <= 0.16 and maximum
absolute error <= 1/256 + 0.16 * maximum absolute reference. Coverage is
**2520 sampled complete dots of 5242880 outputs**, including every N128 tile seam
and token/vector/warp/M128/M256 boundaries. This is not a full-output FP64 oracle.
Every candidate and the header baseline must additionally match all public-route
output BF16 bits, and all outputs must be finite. A production oracle failure
aborts the experiment; it never selects another seed or relaxes the criterion.

Qualification checks eager calls and graph replay, input/weight/prepared-code/
scale/descriptor immutability, guards, and the public workspace's restored cursor
and exact peak. Captured bindings remain fixture-owned until all graphs retire.
Both residual and identity output reset occur before the measured event interval;
residual never accumulates across repetitions. The numerical fixture also checks
the final outputs left by timing before any fresh replay can overwrite them.

Measurements rotate paired route order, with warmed inputs and a separate 128 MiB
read/write scrub **outside** the event interval. Event latency includes only the
TMA GEMM and its epilogue. Quantization, descriptor preparation, output reset,
scrubbing and verification are excluded. Reported whole-graph wall time includes
reset and, in the scrubbed condition, the scrub, so it is not an alternative GEMM
latency. The graph is a timing fixture; shipping prompt processing remains eager.
The output records schedule resources and paired ratios, plus raw samples. Run
one device at a time on an otherwise idle host. A microbenchmark win would still
require a scoped production experiment and complete prompt-processing measurement.
Both local GPUs passed all stated eager/captured numerical, full-output identity,
guard and input-preservation checks for the original three min1 schedules. Across seeds, epilogues and cache conditions,
paired median kernel latency increased 0.9-7.0% for M128/S3/min1 and 5.4-11.2% for
M128/S2/min1. All three schedules reported one resident CTA/SM; the smaller tile's
shared-memory reduction did not produce higher occupancy. **Production retains
M256/S3/min1.** These negative kernel results do not establish a prompt-throughput
gain. [Qualification and measurements](../diagnostics/nvfp4-down-tma-validation.json).


The fourth route changes only M128/S2's minimum-CTA launch bound from one to two.
Its 288-thread CTA has one producer warp, so the M256-only `setmaxnreg` branches
are absent. Two CTAs fit the shared-memory budget, but the observed min1 register
count (139-141) permits only one resident CTA. The CUDA occupancy model predicts
that this nine-warp block needs at most 96 registers/thread to reach two CTAs on
these GPUs; 112 or 128 registers is insufficient because of register-partition
allocation. This is a compiler-pressure experiment, not an occupancy guarantee.

The benchmark reports compiled registers, local bytes per thread, requested
minimum CTAs and occupancy-API CTAs. Target-local ptxas verbose output records the
stack frame and spill stores/loads during compilation; local bytes alone do not
measure executed spill traffic. Four-route rotation retains the same fixed seeds,
numerical criterion, full-output equality, residual resets, guards and final-timed
output checks. On both local GPUs the min2 route compiled to 96 registers and
achieved two resident CTAs/SM. Identity used an 8-byte local frame with four bytes
each of reported spill stores/loads; residual reported no spills. All numerical,
full-bit and guard checks passed, but paired kernel latency increased 10.7-20.3%
across the 16 seed/epilogue/cache/device settings. **The min2 route is rejected;
production still uses M256/S3/min1.** The linked evidence records this followup
separately from the original three-route measurements.

## Local TP2 NVFP4 down decode CTA experiment

`ninfer_nvfp4_down_tp2_cta_bench` compares the public four-warp route, the same
private-header instantiation, and an eight-warp candidate for `[5120,8704]`, T4.
Only CTA grouping changes: A16 activation precision, per-row K traversal and
reduction, weight decoding, and both identity/residual epilogues stay fixed.
The benchmark uses RDC ON like production, two fixed dense-input seeds, and a
complete independent FP64 oracle for every one of the 20,480 outputs. Every
candidate output must also equal production BF16 bits. Eager/captured checks
cover guards, preserved inputs and the zero-workspace contract.

```powershell
.\tools\build_windows.ps1 -Action Build -Target ninfer_nvfp4_down_tp2_cta_bench
.\build\windows\bench\ninfer_nvfp4_down_tp2_cta_bench.exe --device 0 --qualify-only
.\build\windows\bench\ninfer_nvfp4_down_tp2_cta_bench.exe --device 0 --samples 31 --warmup 10
# Repeat both commands with --device 1.
```

Timing rotates all three routes through 31 paired samples after ten warmups,
with warm and 128 MiB scrubbed caches. External CUDA Graph events enclose the
kernel only; residual resets and scrubbing precede the interval. Outputs left
by timing are checked before any new replay. Graph wall time is separate.

On both local cards, eight warps was slower in all 16 seed/epilogue/cache/device
settings, by 1.2-19.0% in paired median kernel latency. Both schedules use 96
registers and zero shared/local bytes; the occupancy API permits 20 resident
warps for four-warp CTAs versus 16 for eight-warp CTAs. These are theoretical
resource limits. Four warps remains the production choice; no engine gain is
claimed. [Evidence](../diagnostics/nvfp4-down-cta-validation.json).

## Local eager TP2 transfer pipeline experiment

`ninfer_peer_transfer_pipeline_bench` compares public pinned allreduce with private
two/four-tile copies for exactly 10 MiB per rank. Each candidate owns one extra
D2H stream per GPU, ordering events and guarded pinned buffers. Main streams pull
peer tiles, join all source/host-buffer readers, then use the unchanged full-buffer
BF16 sum. There is no captured or production routing change in this experiment.

```powershell
.\tools\build_windows.ps1 -Action Build -Target ninfer_peer_transfer_pipeline_bench
.\build\windows\bench\ninfer_peer_transfer_pipeline_bench.exe 2 1
.\build\windows\bench\ninfer_peer_transfer_pipeline_bench.exe 32 31
```

The reduced run still executes all numerical/lifetime checks. Full-coordinate
changing inputs prevent repeated tile patterns from hiding wrong-offset copies.
Every output and pinned publication is checked across consecutive calls, delayed
producers, mixed routes, both GPU orders, and pending-owner destruction. Timings
rotate 31 paired batches after eight warmups; all 32 sums in a batch are joined
before elapsed time is recorded. Host enqueue, complete wall and joined device
intervals are separate. Final timed outputs are checked before each reset.

Two tiles reduced median joined latency from 1566/1564 us to 1520/1518 us for
normal/reversed GPU order (about 3% by paired ratio), while increasing host enqueue
time. Four tiles was slightly slower than two and submitted more work. The full
correctness matrix passed, including Compute Sanitizer with stream-ordered race
tracking and zero reported errors. Both GPUs report one asynchronous copy engine
under WDDM; the result does not claim simultaneous bidirectional DMA or an engine
PP improvement. [Evidence](../diagnostics/peer-transfer-pipeline-validation.json).
