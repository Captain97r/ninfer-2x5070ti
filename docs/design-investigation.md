# Qwen3.8-27B on two RTX 5070 Ti cards

Updated 2026-09-22. Native Windows is primary; Linux portability is retained but
has not been executed locally. Optimize one session near 100K context with TP2,
mixed NVFP4 weights and MTP. Layer partitioning is outside this branch's scope.

## Foundation and artifact

The implementation derives from
[ivanov84/ninfer-windows-tp2](https://github.com/ivanov84/ninfer-windows-tp2/tree/windows-tp2),
itself based on Neroued/ninfer, wamansou/ninfer-tp2-1m and natpate/ninfer-windows.
Source identity is recorded in [engine-source.json](../engine-source.json).
LICENSE, NOTICE and component provenance remain intact. The public fork is
[Captain97r/ninfer-2x5070ti](https://github.com/Captain97r/ninfer-2x5070ti), configured
as `origin`; the Windows TP2 source is `upstream`. Custom changes are maintained
on `main`, with the engine and supporting tools in this repository.

The downloaded primary artifact is a pinned **NInfer v2** Qwen3.8-27B NVFP4
package, 21,492,695,040 bytes, with MTP, proposal head and frontend embedded.
Its full SHA-256 was verified. See [model manifest](../config/model.json).
Current upstream v3 files and existing local GGUFs are not compatible with this
reader. Keep models in `C:\LLM`; runtime weight conversion is not added.

## Hardware and execution

Both cards have 70 SMs, compute capability 12.0, 48 MiB L2 and 16 GB VRAM. The host
has a Ryzen 7800X3D and 32 GB RAM. Active PCIe links are Gen5 x8/x4. Direct CUDA peer
access is unavailable in both directions under this Windows/WDDM configuration.
[Measured hardware](../diagnostics/hardware_report.json)

TP2 divides projections, full-attention KV, GDN recurrent heads, convolution
channels and speculative replay state. Residual activations are replicated at
reduction boundaries. Captured decode exchanges BF16 partials through mapped
pinned-host mailboxes; eager prefill retains staged transfers. The topology is a
communication constraint, not a reason for a2:1 compute split or layer partition.

The native build uses MSVC 2022, nvcc 13.3.73, Ninja, Python 3.11 and the fork's
pinned vcpkg manifest. A UTF-8 console fixes CMake/Ninja interpretation of Russian
MSVC `/showIncludes`; C/C++ source encoding is explicit. This matters for correct
incremental header rebuilding, not just readable logs.

## Implemented correctness and performance work

Mailbox timeout status is checked after both streams retire and before consuming
round output. The final round is covered. A dual-device graph test injects the
fault report from either GPU without creating a real hanging kernel.

Text suffix prefix reuse now rebuilds the missing MTP KV column from retained
hidden state on both ranks. Rewind restores both GDN state shards. There is one
10 KiB hidden transfer per resumed request, outside capture; no per-token duplicate
persistent hidden store. Zero-suffix and multimodal reuse retain full-prefill
fallbacks. Exact matched-schedule append and rolling checkpoint tests pass with
both eager execution and graphs, including 150 peer-egress rounds without mismatch.
[Prefix validation](../diagnostics/prefix-validation.json)

A cold full prefill can use different activation precision and chunk shapes from
retained small-token decode history. Arbitrary cached/cold greedy equality is
therefore not a valid universal contract. The regression preserves the original
different-schedule diagnostic and checks matching computation schedules exactly;
it does not hide numerical drift by choosing an unexplained tolerance.

The new attention specialization targets 70-SM SM120 devices, one 12Q/2KV TP2
shard, INT8 cache and 1/4/5 query tokens at 81,920-131,077 visible keys. It uses 70 active
splits instead of 170 in that interval, while retaining safe broad-graph workspace
and page staging bounds. Other shapes and devices retain the existing selection.
Independent FP64 qualification and actual graph timing govern adoption; kernel
speedups are not presented as whole-model speedups.

The benchmark now accepts and validates TP/device order, assigns EngineOptions
accordingly, and reports both device identities. Its memory fields remain rank 0
measurements, explicitly documented. Correctness tests no longer impose a
portable 100 microsecond budget on host-staged all-reduce; an explicit environment
budget enables hardware-specific timing acceptance.

## Current evidence and remaining work

The CLI has executed NVFP4+TP2+MTP3 with CUDA Graphs. A short 21-token prompt and
64-token output reported 124.41 decode tokens/s; this is a smoke result.
[Smoke record](../diagnostics/smoke.json)

The first 100K benchmark completed at 2,232 prompt tokens/s and 160.07 decode tokens/s.
It cycles the supplied 65,536-token fixture and achieved 94% MTP acceptance, with
one measured repetition. It establishes a working capacity/performance baseline,
not a claim of improvement over the user's separate llama.cpp conversation.
[Initial 100K report](../diagnostics/baseline-100k.json)

The tuned build completed the same single-run workload at 165.07 decode tokens/s
and 2,232 prompt tokens/s, with the same 94% MTP acceptance. Treat this as a
regression check rather than a statistically established end-to-end speedup.
Actual broad-graph attention kernels improved 13-22% on both GPUs.
[Tuned 100K report](../diagnostics/tuned-100k.json),
[Kernel qualification](../diagnostics/gqa-tuning.json)

The user's Q6_K + MTP 90-100 tokens/s remains an unreplicated comparison point. Fair
comparison still needs the same prompts, context, sampler, output length and
acceptance reporting. Further work includes long-session warm-prefix latency,
representative quality evaluation, prefill transport tuning, MTP draft-window
selection, longer runs and Linux validation. Future Qwen4 support requires its
actual model architecture and artifact contract; it is not assumed.
