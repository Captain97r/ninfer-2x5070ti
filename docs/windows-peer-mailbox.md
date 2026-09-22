# The Windows peer-mailbox transport

A technical writeup of the transport added in this fork: why the staged TP2 collective was the
bottleneck on native Windows, how the mailbox works, what it does not solve, and the exact
measurements behind each claim. Written for developers of NInfer and of other single-process
multi-GPU inference engines.

All numbers below come from the benchmark and profile artifacts of this workspace
(`benchmarks/windows-tp2-final.csv`, `benchmarks/windows-tp2-phase2c.csv`, the Nsight Systems
sqlite traces summarized in the phase-2C report) and are specific to one machine:

| Component | Value |
|---|---|
| OS / driver | Windows 11 x64, WDDM, driver 581.57 |
| GPUs | 2 × RTX 5060 Ti 16 GB (`sm_120a`) |
| GPU0 | CPU-attached slot, PCIe 5.0 ×8 (the RTX 5060 Ti is an ×8 card — ×8 is its full link width, not a degraded ×16) |
| GPU1 | Z690 chipset, PCIe 3.0 ×4 (~3.16 GiB/s staging bandwidth, measured) |
| CUDA P2P | `cudaDeviceCanAccessPeer(0,1) == 0` |

---

## 1. Problem

Tensor-parallel decode on two devices needs an allreduce of the partial activations after each
row-parallel projection: 64 layers × 2 projections ≈ **128 collectives per decode round** (129
exchange nodes in the MTP0 graph). Under WDDM on GeForce boards, `cudaDeviceCanAccessPeer`
returns 0, so every collective takes the fork's staged fallback path. Measured on this machine:
**~277 µs per 10 KiB reduction** (reduce_bench), i.e. ~35 ms of a 61 ms MTP0 token — the engine
was communication-bound at **16.37 tok/s** with GPUs idling at 8–12% utilization.

## 2. The original staged allreduce

For each reduction the staged path (from the TP2 base fork) executed:

1. each device copies its partial into a staging buffer (`cudaMemcpyAsync` D2D, transparently
   host-staged by the driver because no P2P was granted);
2. a four-hop cross-device **event chain** orders the two copies and the two reductions;
3. each device reads the peer's partial and produces the sum.

Four WDDM submission round trips and four staged copies for a payload whose *useful* PCIe time is
~3.3 µs at the measured 3.16 GiB/s. The choreography, not the bytes, was the cost — and there
were 128 instances of it per token.

## 3. Why payload bandwidth was not the bottleneck

Direct measurement, not inference:

- the staged one-way bandwidth 0→1 / 1→0 is **3.16 / 3.15 GiB/s** (p2p_probe, 256 MiB), and the
  small-transfer latency is ~70–73 µs per 10 KiB copy — every individual driver-staged copy pays
  a fixed WDDM latency far larger than its transfer time;
- after the mailbox landed, the **40 KiB** exchange costs the same transport floor
  (16.8–19.5 µs) as the **10 KiB** one (15.1–19.1 µs): a 4× payload increase bought no
  measurable latency. A bandwidth-bound transport would have shown a linear increase;
- of the transport floor, pure payload time at Gen3 ×4 is ≈13 µs for 40 KiB — the remaining
  ~4–6 µs is flag round-trip and kernel launch, not transfer.

Conclusion: the PCIe Gen3 ×4 chipset link is **not** saturated by TP decode traffic; it is a
latency/protocol problem.

## 4. WDDM fixed synchronization overhead

On native Windows/WDDM, each cross-device copy submission pays a fixed driver cost (tens of µs)
independent of size, and cross-device event waits serialize the two devices' streams. The staged
protocol amplifies this: 4 staged copies + event hops per collective × 128 collectives per
round. GPU utilization during decode was 8–12% average — the cards were waiting, not
transferring and not computing.

(This is also why the same fork on 2 × RTX 5090 under Linux — where P2P *is* granted — never
showed this profile: with P2P the copies become direct and the event chain is cheap.)

## 5. PeerMailbox design

The replacement keeps the *math* of the collective (an exact BF16 elementwise sum of the two
partials) but changes the *mechanism*: both devices exchange directly through **pinned
write-back host memory**, driven by their own kernels.

Layout (`src/ops/common/peer_mailbox.cu`), one allocation:

```
[ payload rank0 slot0..N-1 ][ payload rank1 slot0..N-1 ]
[ flags rank0 slot0..N-1   ][ flags rank1 slot0..N-1   ][ hang word ]
```

- 2048 slots, each 256-byte aligned (vectorized 16-byte units; distinct cache lines per slot);
  slots are claimed **once per captured call site** at graph-capture time, so every replay fires
  every slot exactly once;
- payload slot size = the widest captured reduction (`hidden × (draft_window+1) × 2` BF16), so
  every decode-round collective fits; anything larger takes the staged path;
- per-slot arrival counters live in each device's VRAM (each rank only touches its own device's
  counter, staying inside one device's memory model).

Protocol (`src/ops/kernel/peer_exchange.cuh`), per exchange, both devices concurrently:

```
rank r kernel:  store partial_r -> host_payload[r]
                __threadfence_system(); __syncthreads();
                flag[r] = 1                        (release)
                poll flag[1-r] == 1 (with __nanosleep backoff, bounded spin)
                read host_payload[1-r]; out_r = partial_r + partial_{1-r}
```

No events, no copy engine, no driver round trip inside the exchange: the critical path is one
PCIe write, one flag round-trip, one PCIe read, all GPU-initiated. The host resets the flags
once per round (plain stores — pinned WB words are coherent with the GPUs' PCIe view), and a
single aggregate **hang word** lets a poller that gave up fault the round loudly instead of
deadlocking (bounded spin < WDDM's 2 s TDR window).

Selection (`src/ops/common/allreduce.cu`, `detail::mailbox_transport`): the mailbox serves a
collective only when

- a PeerMailbox is installed for this ExecutionContext (created once at Program setup when
  `--tp 2` + CUDA graphs),
- `NINFER_TP2_MAILBOX` != 0 (the A/B and safety escape hatch),
- the caller's stream is **capturing** (the decode graph; eager calls keep the staged path so
  flags never dirty a capture),
- the payload is a whole number of 16-byte vectors and fits one slot.

A failed predicate uses the event-ordered path described in section 7.

### Optimized draft selection on this branch

The optimized TP2 draft head uses `argmax_row_parallel` to select a global row before
`proposal_remap_token_ids` maps it to a token. Each rank first reduces its own BF16 logits.
A captured call sends one 16-byte value/index candidate per column from rank 1 to rank 0
through an existing mailbox slot; it does not gather the 131,072-row draft vocabulary.

Only rank 0 waits for the release flag and consumes this candidate. Rank 1 needs no
acknowledgment: the captured call owns its host slot until the whole round retires, and the
receiver never reads rank 1's reusable device scratch. Eager calls, disabled mailboxes,
oversized payloads and exhausted slots use a one-way event-ordered staged copy. Its
read-completion event protects rank 1's scratch before the next call reuses it.

The sum and argmax kernels share the same bounded flag-wait primitive and fault word.
This changes neither the target-logit path nor the MTP acceptance algorithm. The historical
5060 Ti measurements in this document do not measure this draft-selection implementation.

## 6. CUDA graph integration

The whole round — both devices' kernels, the 128 exchanges, the MTP verify/draft branches —
lives in **one cross-device CUDA graph** (fork/join bridge events at the edges; ~1027 graph
nodes per device per MTP4 replay). The exchange kernels are ordinary kernels to the capturer:
capture claims each call site's slot (a plain host-side counter during capture) and bakes the
slot's host addresses into the recorded launches. Between replays the engine does, on the host:

1. read the hang word once — a set bit means a poller gave up last round and the round's
   reductions were skipped: hard fault, not a retry;
2. `PeerMailbox::reset_host_flags()` — plain stores, both GPUs quiesced.

Slot exhaustion (a future topology capturing more call sites than 2048) is reported loudly and
degrades those collectives to the staged path.

## 7. Eager prefill and fallback transport

This branch owns one `PeerTransfer` per Program and stream pair. It contains the ordering
events and, when setup cannot enable P2P in both directions, two `cudaHostAllocPortable`
buffers. Their per-rank capacity is the checked `[hidden, min(prefill_chunk, capacity)]`
BF16 layout: 10 MiB per rank for the 5120-wide, 1024-token prefill configuration.

Eager sum and row-gather calls use these buffers when the largest source is at least 64 KiB
and both sources fit. Both streams must be outside capture. Each source stream copies its
own input to host and records readiness; each destination waits for the peer, copies the
peer's host buffer to local storage, and records read completion. Both streams finally
wait for the peer's read completion before reusing source or host storage. The BF16 sum
uses the existing FP32 add and one BF16 store. There is no compression or model-math change.

The resource is passed explicitly through the projection and collective APIs; the Ops
allocate nothing. Programs retire both streams and destroy graph users before the transfer
resource releases its memory. Distinct Programs have distinct host buffers and events.

Small or over-capacity eager calls retain CUDA-managed cross-device copies. Captured calls
retain the mailbox selection and its existing implicit-copy fallback; the explicit pinned
buffers are never captured. P2P-enabled Programs allocate no host staging. CUDA APIs with
`Async` in their names can still block the host, so performance qualification records both
host enqueue and complete wall time.

`ninfer_peer_transfer_test` checks exact stored sums against a represented-input FP64
oracle, exact asymmetric gathers, changing consecutive inputs without host synchronization,
reversed ranks, two independent owners, device/host guards, and captured fallback. It also
checks pinned buffer contents to establish that the explicit route executed.
`ninfer_peer_transfer_bench` compares implicit and explicit staging with alternating paired
batches; see [the benchmark instructions](../bench/README.md). The historical 5060 Ti
numbers elsewhere in this document do not measure this eager-prefill implementation.

## 8. Correctness validation

- `tools/tp2/mailbox_probe.cu`: mailbox exchange vs the staged path — **exact BF16 sum
  equality**, eager and captured, both payload sizes;
- engine validators: zero "invalid row metadata / licensed tokens" events over all runs;
- repeated 512-token runs: bit-identical output text, identical acceptance (53.06% three times
  for MTP4@512);
- 1024-token run: prefix-identical to the 512-token run up to the cut point;
- 2048-token final run: 586 MTP rounds, **zero graph failures, zero fallback events**, coherent
  greedy text that continues the same sequence;
- a Phase-2C experiment (poll backoff 100→400 ns) was **reverted**, the binary rebuilt, and the
  post-revert control run reproduced the base numbers (68.68 tok/s) with bit-identical text —
  the published numbers are from the canonical tree, not an experimental one.

"Exact BF16 sum" means the mailbox computes the same reduction the staged path computes,
compared operand-for-operand in the probe; it does not claim a different or stronger equivalence
than that.

## 9. Benchmark results

Single request, greedy, thinking off, `--ignore-eos`, 19-token prompt, engine-committed decode
tok/s (full methodology in the README):

| Stage | MTP0 | MTP3 | MTP4 |
|---|---:|---:|---:|
| staged transport (Phase 2A) | 16.37 | 32.58 | — |
| mailbox (Phase 2B) | 35.77 | 63.19 | — |
| mailbox + Phase-2C tuning | 35.73–35.77 | 66.71–66.86 | **68.53–68.87** @512 |
| long runs | — | — | 70.57–70.74 @1024 · **76.65** @2048 |

Phase-2C MTP sweep @512: MTP1 57.71 · MTP2 63.11–63.18 · MTP3 66.71–66.86 · MTP4 68.53–68.87 ·
MTP5 60.82 (rejected: acceptance drops to 40.5%). The 2048-token run reached 62.33% acceptance
(3.49 tokens/round) on repetitive markdown — that figure is workload-dependent and must not be
quoted as a general speed.

Communication microbenchmarks (per exchange pair, medians):

| Exchange | dev0 / dev1 median | transport floor | lockstep excess |
|---|---:|---:|---:|
| 10 KiB (MTP0 shape) | 17.3 / 19.3 µs | 15.1 / 19.1 µs | ~0 |
| 40 KiB (MTP4 shape) | 59.3 / 65.3 µs | 16.8 / 19.5 µs | **42.5 / 45.9 µs** |

## 10. Remaining bottlenecks (MTP4 round, ~45.5 ms)

| Category | share of round |
|---|---:|
| NVFP4/W8 GEMM (memory-bound, ~95% of practical VRAM bandwidth) | ~60% |
| communication (mailbox exchanges) | ~18% |
| in-graph lockstep gaps (waiting for the peer) | ~11% |
| host between graph replays | ~4% |
| lm_head | ~3% |
| GDN | ~2% |
| attention | <1% |

- **70% of the MTP4 communication budget is lockstep waiting**, not transport: the side that
  arrives first at an exchange spins on the peer's flag. In MTP0 (uniform round) the excess is
  ~0; in MTP4 the batch-5 round has asymmetric segment durations on the two cards (including a
  measured dev0 display-GPU penalty of ~1.7 ms/round with HAGS off).
- Matmul is at the practical hardware ceiling for NVFP4 weights; the only lever left is reading
  weights fewer times per committed token (which MTP already does).
- No legal overlap exists inside the round on this architecture: every allreduce's output is
  consumed replicated by the next projection (KV and GDN state are replicated on both devices by
  the base fork's design). Reduce-scatter/sequence-parallelism would change that contract and
  is future work.

The engine has moved from **communication-bound** to **memory-bandwidth/synchronization-bound**.

## 11. Applicability and limitations

Where the mailbox is the right tool:

- two GPUs in one process, WDDM/GeForce (no P2P), CUDA-graph decode, small fixed-size
  collectives dominating the round — exactly the RTX 5060 Ti class machine tested here.

Where it is not:

- P2P-capable systems (Linux, or pro cards under TCC): a direct peer path has lower latency than
  any host round trip; the mailbox is **not** installed by hardware condition in this fork — it
  installs whenever `--tp 2` + CUDA graphs are on — so use `NINFER_TP2_MAILBOX=0` there, or gate
  the install on `cudaDeviceCanAccessPeer` before adopting (see README Limitations);
- bandwidth-heavy regimes (prefill, multi-request batches): the staged path is retained for
  those on purpose;
- this is a **fallback-transport optimization**, not a replacement for NCCL or any claim that
  Windows beats Linux: no Linux comparison was measured on identical hardware. The single
  external reference point (a 2 × 5060 Ti Linux/vLLM 122k-context community result, 67.293
  tok/s decode) is a different serving stack, context length, and KV format, and is quoted only
  as orientation.

Tested on one machine, one driver, one model family (Qwen3.8-27B NVFP4). No claim of
portability is made beyond it.
