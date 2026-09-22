# NInfer for 2 x RTX 5070 Ti

Native Windows inference for **Qwen3.8-27B mixed NVFP4, tensor parallelism and MTP**
on two 16 GB RTX 5070 Ti GPUs. Linux code paths are retained; Linux execution has
not been validated locally. Models live in **C:\LLM**, outside this checkout.

## Run

Clone this fork and follow the build steps below. The native CLI and server are
built in `build/windows/apps/`. From the repository root:

```powershell
# One prompt; defaults to TP2, INT8 KV, MTP3, vision and 199,680-token capacity.
.\tools\run_windows.ps1 -Prompt 'Explain tensor parallel inference.'

# OpenAI/Anthropic-compatible server with image input, localhost:8000, one active request.
.\tools\run_windows.ps1 -Mode Server
```

Use `-Context 4096` for a small allocation, `-DraftTokens 0` for ordinary decode,
`-DraftTokens 4` to evaluate a larger draft window, or `-NoCudaGraph` for eager
execution. Use `-NoThinking` to disable thinking by default in either mode. CLI sampling follows the artifact defaults; `-Greedy -NoThinking`
reproduces the short smoke-run settings. Server requests control their own sampling.

The model-discovery endpoints publish the configured context window as `max_model_len`
and `context_length`, so clients such as oh-my-pi can budget the session correctly.
The default is **199,680 tokens**, shared by prompt, image and generated tokens;
changing `-Context` changes the advertised limit too.
When a client omits its output limit, the Windows server launcher allows generation
up to the remaining context instead of cutting replies at 8,192 tokens. Explicit
client output limits still apply. OMP treats a `length` finish as an incomplete
reply and can compact even when the context window has room.

After updating an existing OMP setup, start the server, open `/models` in OMP,
select the vLLM provider and press F5 to refresh. Reopen OMP so its cached model
metadata reflects the new limit.

The pinned model is
`C:\LLM\neroued\Qwen3.8-27B-nvfp4-NInfer-v2\qwen3_8_27b_nvfp4.ninfer`.
Its MTP weights and tokenizer are embedded. [config/model.json](config/model.json)
records the exact compatible revision, size and SHA-256. Current upstream **v3**
artifacts and GGUF files cannot be substituted for this v2 artifact.

## Image input

Vision is enabled by the Windows launcher in both server and CLI modes. The pinned
artifact already contains the vision encoder. Use `-NoVision` for a text-only
allocation. The default `-ImageMaxTokens 2048` budget allows approximately 2 MP per
image and automatically downsizes larger images while preserving aspect ratio;
a larger budget uses more GPU memory. Image tokens count toward the context window.

OpenAI requests carry images in `image_url` content parts, using an HTTP(S) URL or
a base64 data URL. The server advertises image input only when vision is enabled.
For CLI image requests, use `-MessagesFile` with the engine's multimodal message
format; see [serving details](docs/serving.md) and
[CLI examples](examples/cli/messages/image_chart.json).

OMP 18.2.8's built-in vLLM provider ignores discovered image capabilities. Its
`~/.omp/agent/models.yml` therefore needs this per-model override (installed locally):

```yaml
providers:
  vllm:
    modelOverrides:
      qwen3.8-27b:
        input: [text, image]
```

After starting the server, restart OMP so it reads this configuration. You can then
attach an image to the Qwen session. This override preserves context discovery and
provider credentials. With `-NoVision`, only send text requests.

To repeat the bounded image-grounding checks against an already running server:

```powershell
py -3.11 .\tools\test_vision.py
```

[Local vision validation](diagnostics/vision-validation.json) passed seven checks: OCR,
counting and position, streamed answers, changed images, two-image history, thinking,
large-image downscaling, and text after image requests. A real OMP attachment also
returned the correct answer. These were short image prompts at 102,400-token server
capacity with TP2, MTP3 and CUDA Graphs. The later
[workspace validation](diagnostics/tp2-workspace-validation.json) passed eight
image/text checks at 199,680-token capacity, including the full 2,048-image-token
budget. A full 199K multimodal history has not been tested.

## Build and validate

Requires Visual Studio C++ x64 Build Tools, CUDA 12.8+ (tested with nvcc 13.3.73),
CMake 3.28+, Ninja, Python 3.11 and vcpkg. The build helper finds standalone Visual
Studio Build Tools and its bundled vcpkg, uses the manifest's pinned dependencies,
and handles localized compiler output. Pass `-VcpkgRoot` if vcpkg lives elsewhere.

```powershell
.\tools\download_model.ps1
.\tools\build_windows.ps1 -Tests -Benchmarks
.\tools\test_windows.ps1 -Model
```

Downloads stream into a resumable `.part` file and verify size, format and SHA-256
before a non-overwriting rename. Existing files are preserved. Use
`download_model.ps1 -ValidateOnly` for an explicit full checksum recheck.

`test_windows.ps1` builds and serially runs the focused host and GPU checks:
transport, exact distributed draft selection, packed gather, NVFP4/TMA descriptor
lifetime, sampling, speculative rounds, TP/MTP and 70-SM attention. `-Model` adds
the real-artifact prefix regression. It does not run the inherited TP1 reference
tests, whose full weights exceed one 16 GB card. Build with `-Fresh` once when
changing compiler detection/console settings.

Run response-quality checks separately. These commands each start and stop their
own temporary TP2/MTP3 server with vision and 102,400-token capacity:

```powershell
py -3.11 .\tools\run_quality_windows.py --label quality-check --baseline diagnostics/quality-reference.json
py -3.11 .\tools\run_quality_windows.py --label retrieval-check --fixtures tests/data/retrieval-panel.json --baseline diagnostics/retrieval-reference.json
```

Use a new report label for each run. Reports separate exact observable response
parity from task accuracy: the saved short panel scores 16/18, retaining two task
failures, and the four synthetic retrieval cases score 4/4 at approximately
4K, 32K and 100K tokens. These are bounded regression checks; see the
[quality criteria and limitations](docs/design-investigation.md#quality-qualification-implemented-checks-and-remaining-limits).

A fresh build from the consolidated repository also passed the media decode and
OpenAI/Responses schema tests, plus both application entry points. The inherited
`ninfer_qwen3_6_frontend_test` still exits with `0xC0000409` on this Windows host;
the [base fork records the same failure](https://github.com/ivanov84/ninfer-windows-tp2/commit/2ea110eeee606e24d9383fc8e719d4cc04ec8bf0).

## Implemented changes

- TP2+MTP text suffix reuse, including both-rank GDN checkpoint restoration and
  reconstruction of the missing MTP KV column from retained hidden state.
- Mailbox failure detection at round completion, before consuming output, including
  the final decode round.
- A 70-SM SM120 INT8 attention schedule for one TP2 shard, 1/4/5 token queries and
  81,920..200,709 visible keys. Graphs retain safe workspace/page bounds.
- Windows NVFP4 TMA descriptor lifetime and visibility fixes, including release
  on the owning stream and host descriptors retained by captured graphs.
- Exact distributed MTP draft argmax with compact candidate transfer, preserving
  global tie/NaN handling, padding exclusion and token-ID mapping.
- Explicit pinned-host staging for large eager TP2 prefill collectives, preserving
  BF16 arithmetic and the captured transport path. Full 10 MiB reductions use
  two copy tiles on the qualified Windows dual-5070-Ti profile.
- Exact packed target-logit gathering with explicitly planned peer scratch;
  one-token gathering and optimized draft selection retain their existing paths.
- Captured TP2 MTP acceptance on rank zero, with exact compact decision transfer
  and peer-local penalty-counter updates; eager acceptance remains replicated.
- Fixed HTTP quality and long-context retrieval fixtures, with prompt-count checks
  and exact response comparison separate from task scoring.
- TP2 workspace sized from shard dimensions and actual phase lifetimes, saving
  60.63 MiB per GPU with the default vision budget.
- Cold long-code/document sweeps, sustained generation measurements, and exact
  near-limit retrieval and context-exhaustion qualification.
- Actual TP2 options and both device identities in the end-to-end benchmark.
- Reproducible Windows build, model download, launch and focused test scripts.

Zero-suffix and multimodal prefix reuse still use full prefill. Cached and cold
runs can differ numerically when they process the same history with different
activation precision or chunk boundaries. The prefix regression checks matched
schedules exactly, plus reproducible retained decode history and both-rank egress;
it does not promise arbitrary cached/cold greedy output identity.

## Validation and measurements

The current Windows default is **199,680 tokens** with MTP3 and the 2,048-token
vision budget retained. Measured memory use is approximately **14.6 GiB per GPU**.
Moving the display to the motherboard changed same-configuration throughput by
less than 0.2%; the useful changes are tighter workspace allocation and attention
tuning for the enlarged context.

Cold code/document requests measured approximately **3,140 / 2,324 / 1,873 prompt
tokens/s** at 8K / 100K / 180K prompts. Generation measured **145-165 / 130-149 /
118-134 tokens/s**, with natural replies of about 4K-7K tokens. These are single
requests per workload, not guaranteed rates. The 180K replies changed with the
attention reduction schedule, so their before/after throughput is not an isolated
same-output speedup. [Long-context results and limits](docs/performance.md#local-dual-5070-ti-long-context-profile).

A separate sustained run generated **32,768 tokens at 124.15 tok/s**, with stable
measured memory. It reached the output cap while reasoning, so its math answer is
unscored; its actual context reached 33K.

All **29 saved response observables** still match exactly; inherited task scores
remain 16/18, 5/7 and 4/4. Near-limit retrieval, exact context exhaustion and recovery
pass. Attention passes independent FP64 checks on both GPUs, including the new
context boundary. These are bounded correctness and regression results, not a
claim of unchanged answers for every long conversation.

The initial [focused test run](diagnostics/validation.json) passed **11/11 checks**,
including the real-model prefix regression, before the additional operator tests
were added. Subsequent stage records cover [TMA descriptor ownership](diagnostics/tma-descriptor-validation.json),
[exact draft selection](diagnostics/draft-argmax-validation.json), and
[bulk prefill transfer](diagnostics/bulk-transfer-validation.json), including their
operator, response-quality and workload-specific performance results.

The packed target-logit build passed the updated **16/16 focused checks**, the
separate real-model prefix regression, and exact response comparison on all 18
short-panel and four retrieval cases. Its task scores remain 16/18 and 4/4.
Against the bulk-transfer build, synthetic-corpus generation increased from
199.52 to **205.17 tokens/s at 8K** and from 172.45 to **176.91 tokens/s at 100K**
(2.83% and 2.58%; three measured repetitions after one warmup). Prompt throughput
was effectively unchanged. These are workload-specific results, not ordinary
coding-session throughput. [Runtime evidence](diagnostics/column-gather-runtime-validation.json)
and [measurement details](docs/performance.md#local-dual-5070-ti-packed-target-logit-gather).

Captured rank-zero acceptance adds **0.73% / 0.59% TG** in a fresh matched comparison,
reaching **206.61 / 177.92 tokens/s** at 8K / 100K on that synthetic corpus.
That build passed **18/18** focused checks, and all **29**
short, stochastic and retrieval response observables match their saved references.
Baseline task scores remain 16/18, 5/7 and 4/4; these checks do not claim perfect
model accuracy. [Runtime evidence](diagnostics/rank0-acceptance-runtime-validation.json)
and [measurement details](docs/performance.md#local-dual-5070-ti-captured-rank-zero-acceptance).

The two-tile prompt-transfer stage passed **19/19** focused checks and
preserved all **29** saved response observables. Against the preceding runtime,
prompt processing increased from 3138.39 to **3206.52 tok/s at 8K** and from
2289.11 to **2324.17 tok/s at 100K** (**2.17% / 1.53%**). Generation is unchanged
at approximately **206.76 / 178.01 tok/s** on that synthetic corpus. No weight,
activation or KV precision changes were made. [Runtime evidence](diagnostics/peer-transfer-pipeline-runtime-validation.json)
and [measurement details](docs/performance.md#local-dual-5070-ti-two-tile-prompt-transfers).

A separate short-prompt serving comparison against the original local engine
measured **149.1 tok/s for Python, 117.3 for prose and 205.3 for structured output**,
about **6.4-6.6% faster**, with all nine response observables and MTP counters
unchanged. Eight responses hit the output cap; one Python response ends early
with tool-like text. These are throughput/parity samples, not completed-task
scores. [Method and limitations](docs/performance.md#local-dual-5070-ti-short-prompt-serving-comparison).

The [localhost server smoke test](diagnostics/server-smoke.json) returned a valid
chat response with 102,400-token capacity; the test process was stopped afterward.

[smoke.json](diagnostics/smoke.json) records a successful native NVFP4+TP2+MTP3
CLI run: 21 prompt tokens, 64 generated tokens, 124.41 reported decode tokens/s.
This short run is a functional smoke check.

[baseline-100k.json](diagnostics/baseline-100k.json) records the first untuned
100K run: 2,232 prompt tokens/s and 160.07 generated tokens/s, one measured
repetition. The input cycles the supplied 65,536-token benchmark fixture to
100,000 tokens and has **94% MTP acceptance**. It is a synthetic capacity/performance
check, not a comparison against the user's Q6_K conversation or a quality result.

[tuned-100k.json](diagnostics/tuned-100k.json) records the same workload after the
attention change: 2,232 prompt tokens/s and **165.07 generated tokens/s**, again
with 94% MTP acceptance. These single runs show no throughput regression; they
are not a statistical speedup estimate. The qualified isolated attention kernels
are 13-22% faster at 100K on both cards, measured with production compilation and
broad CUDA Graph envelopes. [Attention evidence](diagnostics/gqa-tuning.json)

[Prefix validation](diagnostics/prefix-validation.json) covers exact matched-schedule
append and rolling checkpoint restoration under eager execution and graphs, with
150 speculative rounds and zero cross-rank disagreements.

The [hardware report](diagnostics/hardware_report.json) confirms two 70-SM SM120
GPUs, active Gen5 x8/x4 links and no direct CUDA peer access under Windows WDDM.
The original [GPU probe](tools/README.md) remains independently reproducible.

## Source

This is [Captain97r/ninfer-2x5070ti](https://github.com/Captain97r/ninfer-2x5070ti),
maintained on `main`. It preserves the history of
[ivanov84/ninfer-windows-tp2](https://github.com/ivanov84/ninfer-windows-tp2/tree/windows-tp2),
which combines [Neroued/ninfer](https://github.com/Neroued/ninfer),
[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) and the native
Windows port from [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows).
The base revision is recorded in [engine-source.json](engine-source.json).
[LICENSE](LICENSE), [NOTICE](NOTICE) and component provenance are preserved.

The inherited foundation supplies NVFP4 execution, TP2, MTP and Windows support;
the changes listed above specialize and validate it for this configuration.
See the [design notes](docs/design-investigation.md), [engine documentation](docs/README.md)
and [inherited Windows guide](https://github.com/ivanov84/ninfer-windows-tp2/blob/2ea110eeee606e24d9383fc8e719d4cc04ec8bf0/README.md).
`origin` points to this fork; `upstream` points to the Windows TP2 source.
