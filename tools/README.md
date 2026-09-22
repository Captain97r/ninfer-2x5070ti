# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
The Windows helpers below provide the download-and-run path described in the
[project README](../README.md); the remaining tools are maintainer workflows.

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Windows helpers

- `download_model.ps1`: download or verify the pinned artifact under `C:\LLM`.
- `build_windows.ps1`: configure and build the native applications and selected tests.
- `run_windows.ps1`: launch the CLI or server (port 8000, TP2, MTP3, vision, 102,400 tokens).
- `test_windows.ps1`: build and run focused checks; `-Model` includes the real prefix test.
- `test_vision.py`: exercise image grounding against an already running server.
- `run_gpu_probe.ps1`: measure the local hardware and interconnect as described below.

## Task index

| Task | Location |
|---|---|
| Build the 27B artifact | [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Build the Qwen3.8-27B artifact | [`convert/qwen3_8_27b/`](convert/qwen3_8_27b/) |
| Build the 35B-A3B artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The converters consume an official local BF16 checkpoint and write one complete `.ninfer`
artifact. The paths below are placeholders for the maintainer's local checkpoint checkouts:

```bash
python3 -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect either result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

The exact source revisions, inventories, formats, and conversion recipes are recorded in
[`docs/maintainer/`](../docs/maintainer/). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine. See the parity README for the direct 27B
artifact/source Vision comparison command.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```

# GPU/interconnect probe

From the repository root on Windows, with Visual Studio 2022 C++ Build Tools and
the CUDA Toolkit installed:

```powershell
.\tools\run_gpu_probe.ps1
```

The launcher uses `CUDA_PATH` if defined, otherwise CUDA 13.3. Override with
`-CudaPath 'C:\path\to\CUDA'`. It writes a combined hardware inventory and
measurements to `diagnostics/hardware_report.json`, raw native measurements to
`diagnostics/gpu_probe.json`, and a temporary build to `diagnostics/build/`.
It does not install dependencies or download model weights.

The C++ source also uses portable CUDA runtime APIs and can be compiled on Linux:

```sh
mkdir -p diagnostics/build
g++ -O2 -std=c++17 tools/gpu_probe.cpp -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
  -o diagnostics/build/gpu_probe
diagnostics/build/gpu_probe > diagnostics/gpu_probe.json
```

The Linux command has not been executed on this Windows host. It produces the
native probe JSON; the Windows launcher additionally records CPU, RAM,
motherboard, and `nvidia-smi` PCIe samples.

## Measurement method

- Query all visible devices and both directions of `cudaDeviceCanAccessPeer`.
- Allocate one 16 MiB device buffer on each of CUDA ordinals 0 and 1 and one
  16 MiB portable pinned host buffer.
- For each direction and size (8 KiB, 10 KiB, 64 KiB, 1 MiB, 16 MiB), warm up
  eight times, then time 100 iterations (40 for 16 MiB).
- Each iteration performs D2H, waits for completion, then H2D and waits again.
  Host wall time includes device switching, API calls, and both waits.
  Initial source upload and destination clearing are explicitly synchronized
  before the nonblocking transfer streams start.
- Compare the destination with a deterministic, nonuniform byte pattern after
  each measurement. A failed comparison or CUDA call makes the probe fail.
- Run a separate three-second copy load so the launcher can observe active
  PCIe generation and width. Those copies are excluded from transfer timings.
  Eight warmup iterations do not establish sustained GPU clocks; the three-second
  load happens after all measured transfer tests.

The 10 KiB case represents one 5,120-element BF16/FP16 hidden-state vector.
Effective GB/s counts the payload once even though host staging transfers it
across PCIe twice. This serial baseline does not implement double buffering,
overlap, collective operations, or an inference workload. Median, mean, and p95
are included because Windows WDDM scheduling can introduce substantial jitter.
Concurrent `nvidia-smi` sampling also contributes some system overhead.

Memory readings are snapshots, not guarantees of allocatable VRAM. In the
recorded Windows run, `cudaMemGetInfo` and `nvidia-smi` disagreed significantly on
GPU 0; preserve both readings and budget conservatively. No large allocation
test is performed. Timing results are measurements of the current machine and
software configuration, not predicted model token rates.

## Recorded run

The report dated 2026-09-22 was built with CUDA 13.3 and MSVC on Windows, using
NVIDIA driver 616.92. Both cards report SM 12.0, 70 multiprocessors, 100 KiB shared
memory per SM, 99 KiB opt-in shared memory per block, and 48 MiB L2.

CUDA peer access is unavailable in both directions. Under the short copy load,
GPU 0 reports PCIe Gen5 x8 and GPU 1 reports Gen5 x4. The cards' Gen5 x16 maximum
capability is separate from the negotiated motherboard links.

| Payload | 0 to 1 median | 1 to 0 median | 0 to 1 effective GB/s | 1 to 0 effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| 8 KiB | 41.00 us | 32.20 us | 0.180 | 0.249 |
| 10 KiB | 41.10 us | 32.40 us | 0.227 | 0.309 |
| 64 KiB | 75.45 us | 75.50 us | 0.875 | 0.926 |
| 1 MiB | 149.30 us | 175.00 us | 6.631 | 5.905 |
| 16 MiB | 1812.70 us | 1789.45 us | 9.218 | 9.313 |

Bandwidth uses the mean duration; the latency columns use medians. All ten
size/direction checks passed. Results can vary with desktop activity, driver
settings, power state, and other GPU work; rerunning replaces the JSON files.
