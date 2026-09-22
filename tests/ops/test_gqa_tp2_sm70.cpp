// Production dispatch qualification for the measured 70-SM, 12Q/2KV INT8 route.
// Exact eager windows and the broad CUDA Graph envelope used by MTP must both
// evaluate the independent FP64 oracle. A graph also replays at short positions,
// where the original split policy must remain valid. No model artifact is needed.
#include "ninfer/ops/gqa_attention.h"
#include "ops/gqa_attention_fixture.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gqa;

namespace {
constexpr Geometry kGeometry{"tp2-sm70", 12, 2};
constexpr int kContext = 100000;
constexpr int kEnvelopeEnd = 131077;

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};

struct Graph {
    cudaGraph_t definition = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (definition) cudaGraphDestroy(definition);
    }
};

int check_output(GuardedDeviceBuffer& output, const std::vector<double>& reference,
                  const char* label) {
    const auto actual = bf16_bits_to_double(copy_from_guarded<std::uint16_t>(output, reference.size()));
    const auto criterion = long_window_attention_criterion(
        DType::I8, bf16_storage_floor_relative_l2(reference));
    return verify_reduction(label, actual, reference, criterion) + output.verify_guards(label);
}

int run(int device) {
    cuda_check(cudaSetDevice(device), "select device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "device properties");
    if (properties.major != 12 || properties.minor != 0 || properties.multiProcessorCount != 70) {
        std::puts("SKIP: requires a 70-SM sm_120 GPU");
        return 77;
    }
    Stream stream;
    HostCache expected = make_cache(kGeometry, DType::I8, kEnvelopeEnd, 5070);
    const auto q = make_bf16_values(12 * 256 * 5, 17035, -0.25f, 0.25f);
    const auto k = make_bf16_values(2 * 256 * 5, 17036, -0.25f, 0.25f);
    const auto v = make_bf16_values(2 * 256 * 5, 17037, -1.0f, 1.0f);
    std::vector<std::int32_t> positions(5);
    for (int t = 0; t < 5; ++t) positions[t] = kContext + t;
    append_cache(expected, k, v, positions);
    std::fprintf(stderr, "GPU%d: computing full FP64 100K oracle...\n", device);
    const auto full_reference = ideal_attention(q, expected, positions);
    DeviceCache cache(expected, MappingPattern::Fragmented);
    auto dq = to_device_bf16(q), dk = to_device_bf16(k), dv = to_device_bf16(v);
    auto poison_k = to_device_bf16(std::vector<float>(2 * 256 * 5, 0.0f));
    auto poison_v = to_device_bf16(std::vector<float>(2 * 256 * 5, 64.0f));
    auto dp = to_device(positions);
    auto row = to_device(std::vector<std::int32_t>{0});
    auto valid = to_device(std::vector<std::int32_t>{5});
    int failures = 0;
    for (const int tokens : {1, 4, 5}) {
        const std::size_t count = 12 * 256 * tokens;
        const std::vector<double> reference(full_reference.begin(), full_reference.begin() + count);
        Tensor tq(dq.p, DType::BF16, {256, 12, tokens});
        Tensor tk(dk.p, DType::BF16, {256, 2, tokens});
        Tensor tv(dv.p, DType::BF16, {256, 2, tokens});
        Tensor tp(dp.p, DType::I32, {tokens});
        Tensor tpoison_k(poison_k.p, DType::BF16, {256, 2, tokens});
        Tensor tpoison_v(poison_v.p, DType::BF16, {256, 2, tokens});
        Tensor tr(row.p, DType::I32, {1});
        Tensor tc(valid.p, DType::I32, {1});
        GuardedDeviceBuffer output(count * sizeof(std::uint16_t));
        Tensor tout(output.data(), DType::BF16, {256, 12, tokens});
        valid.copy_from_host(&tokens, sizeof(tokens));
        dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
        const std::array<ops::GqaExecutionEnvelope, 2> envelopes{{
            {static_cast<std::uint32_t>(kContext + tokens), static_cast<std::uint32_t>(kContext + tokens)},
            {1, kEnvelopeEnd}}};
        for (const auto envelope : envelopes) {
            const auto bytes = ops::gqa_attention_workspace_capacity_bytes(
                12, DType::I8, envelope, 1, tokens, tokens);
            GuardedDeviceBuffer scratch(bytes);
            WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
            for (bool append : {false, true}) {
                const auto call = [&] {
                    if (append) {
                        ops::gqa_attention(tq, tk, tv, tp, tc, tr, kAttentionScale,
                                           cache.batch_view(), envelope, workspace, tout, stream.value);
                    } else {
                        ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope,
                                                  workspace, tout, stream.value);
                    }
                };
                // Overwrite the new rows with deliberately large V values via the independent
                // append-only route. A missing fused append store must fail the attention
                // oracle, even when the same graph was already replayed successfully.
                const auto poison_append_rows = [&] {
                    if (append) ops::gqa_kv_append(tpoison_k, tpoison_v, tp, cache.view(), stream.value);
                };
                poison_append_rows();
                // This eager call also initializes per-device launch metadata before capture.
                call();
                cuda_synchronize();
                failures += check_output(output, reference, append ? "append exact/eager" : "cached exact/eager");
                Graph graph;
                cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal), "capture start");
                call();
                cuda_check(cudaStreamEndCapture(stream.value, &graph.definition), "capture end");
                cuda_check(cudaGraphInstantiate(&graph.executable, graph.definition, 0), "instantiate graph");
                poison_append_rows();
                cuda_check(cudaGraphLaunch(graph.executable, stream.value), "graph launch");
                cuda_synchronize();
                failures += check_output(output, reference, append ? "append graph" : "cached graph");
                // Real MTP masks the rejected suffix. Reuse this SAME captured append graph.
                if (append && tokens == 5 && envelope.min_visible_keys == 1) {
                    poison_append_rows();
                    cuda_synchronize();
                    const int accepted = 4;
                    valid.copy_from_host(&accepted, sizeof(accepted));
                    auto masked_positions = positions;
                    masked_positions[4] = masked_positions[3];
                    dp.copy_from_host(masked_positions.data(), masked_positions.size() * sizeof(std::int32_t));
                    auto masked_reference = reference;
                    std::fill(masked_reference.begin() + 12 * 256 * 4, masked_reference.end(), 0.0);
                    cuda_check(cudaGraphLaunch(graph.executable, stream.value), "masked replay");
                    cuda_synchronize();
                    failures += check_output(output, masked_reference, "masked MTP graph replay");
                    const auto masked_bits = copy_from_guarded<std::uint16_t>(output, count);
                    if (std::any_of(masked_bits.begin() + 12 * 256 * 4, masked_bits.end(),
                                    [](std::uint16_t bits) { return bits != 0; })) {
                        throw std::runtime_error("masked MTP output is not exact BF16 zero");
                    }
                    valid.copy_from_host(&tokens, sizeof(tokens));
                    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
                    cuda_check(cudaGraphLaunch(graph.executable, stream.value), "restore full append");
                    cuda_synchronize();
                }
                if (!append && envelope.min_visible_keys == 1) {
                    std::vector<std::int32_t> short_positions(tokens);
                    for (int t = 0; t < tokens; ++t) short_positions[t] = 128 + t;
                    const auto short_reference = ideal_attention(q, expected, short_positions);
                    dp.copy_from_host(short_positions.data(), short_positions.size() * sizeof(std::int32_t));
                    cuda_check(cudaGraphLaunch(graph.executable, stream.value), "short-window replay");
                    cuda_synchronize();
                    failures += check_output(output, short_reference, "broad graph short-window replay");
                    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
                }
            }
            failures += scratch.verify_guards("exact queried workspace capacity");
            if (workspace.used() != 0 || workspace.peak_used() != bytes)
                throw std::runtime_error("GQA workspace query/execution mismatch");
        }
    }
    failures += verify_cache("append graph persistent KV codec", cache.snapshot(), expected);
    failures += cache.verify_guards("production TP2 GQA cache");
    std::printf("GPU%d: production TP2 GQA exact/broad/masked graph checks: %s\n", device,
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (cuda_unavailable()) return 77;
        if (argc > 2) throw std::invalid_argument("usage: ninfer_gqa_tp2_sm70_test [device]");
        return run(argc == 2 ? std::stoi(argv[1]) : 0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "TP2 GQA production test failed: %s\n", error.what());
        return 1;
    }
}
