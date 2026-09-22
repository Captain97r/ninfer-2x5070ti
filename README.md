# NInfer for 2 x RTX 5070 Ti

Native Windows inference for **Qwen3.8-27B mixed NVFP4, tensor parallelism and MTP**
on two 16 GB RTX 5070 Ti GPUs. Linux code paths are retained; Linux execution has
not been validated locally. Models live in **C:\LLM**, outside this checkout.

## Run

Clone this fork and follow the build steps below. The native CLI and server are
built in `build/windows/apps/`. From the repository root:

```powershell
# One prompt; defaults to TP2, INT8 KV, MTP3, vision and 102,400-token capacity.
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
The default is 102,400 tokens; changing `-Context` changes the advertised limit too.
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
capacity with TP2, MTP3 and CUDA Graphs; a full 100K multimodal history was not tested.

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

`test_windows.ps1` builds and runs the focused host, transport, NVFP4, TP/MTP and
70-SM attention checks; `-Model` adds the real-artifact prefix regression. It does
not run the inherited TP1 reference tests, whose full weights exceed one 16 GB
card. Build with `-Fresh` once when changing compiler detection/console settings.

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
  approximately 80K-128K visible keys. Graphs retain safe workspace/page bounds.
- Actual TP2 options and both device identities in the end-to-end benchmark.
- Reproducible Windows build, model download, launch and focused test scripts.

Zero-suffix and multimodal prefix reuse still use full prefill. Cached and cold
runs can differ numerically when they process the same history with different
activation precision or chunk boundaries. The prefix regression checks matched
schedules exactly, plus reproducible retained decode history and both-rank egress;
it does not promise arbitrary cached/cold greedy output identity.

## Validation and measurements

The final [focused test run](diagnostics/validation.json) passed **11/11 checks**,
including the real-model prefix regression. The [localhost server smoke test](diagnostics/server-smoke.json)
returned a valid chat response with 102,400-token capacity; the test process was
stopped afterward.

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
