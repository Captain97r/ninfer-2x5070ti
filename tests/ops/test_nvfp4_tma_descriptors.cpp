#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/linear/linear_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

// Registered single-device gate/up shape. The separate TP2 split suite covers the shard
// dispatch. Ordinary and fused projection use DIFFERENT weight-code tile shapes at this
// same [N,K], so alternating them also changes descriptor geometry.
constexpr std::int32_t kN = 34816;
constexpr std::int32_t kK = 5120;
constexpr std::int32_t kT = 1024;
constexpr auto kPolicy = ops::LinearPolicy::AllowA4;
// All 16 logical weight-pattern phases plus registered tile/shard boundary rows.
constexpr std::array<std::int32_t, 30> kRows{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    63, 64, 127, 128, 8703, 8704, 17407, 17408, 17409, 17471, 17472, 34687, 34751, 34815};
constexpr std::array<std::int32_t, 10> kColumns{0, 1, 255, 256, 511, 512, 767, 768, 1022, 1023};

// This test isolates descriptor ownership/lifetime, not the lossy-A4 error envelope.
// For each active K16 block, choose BF16 values q * 2^-e, q from E2M1, e=5..8,
// and include |q|=6. With the fixture's input divisor 3.5, the block scale is
// (7/2)*2^-e: exactly E4M3 7/64, 7/128, 7/256, or 7/512. Thus activation
// quantization is exact and the original-input FP64 oracle can use the stronger
// A16 numerical criteria. General dense lossy-A4 arithmetic remains covered by
// the separate Linear/LinearAdd/LinearSwiGLU operator suites.
std::uint64_t mix_fixture_coordinate(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint16_t> make_lifetime_activation(std::uint32_t seed) {
    constexpr std::int32_t kBlock = 16;
    constexpr std::int32_t kBlocks = kK / kBlock;
    constexpr std::array<float, 8> kMagnitudes{0.0F, 0.5F, 1.0F, 1.5F,
                                              2.0F, 3.0F, 4.0F, 6.0F};
    static_assert(kK % kBlock == 0 && kBlocks == 320);
    std::vector<std::uint16_t> values(static_cast<std::size_t>(kK) * kT, 0);

    // Dense token zero touches every K and mixes signs, E2M1 magnitudes, and scales.
    for (std::int32_t block = 0; block < kBlocks; ++block) {
        const auto block_key = mix_fixture_coordinate(
            (static_cast<std::uint64_t>(seed) << 32) | static_cast<std::uint32_t>(block));
        const int exponent = 5 + static_cast<int>((block_key >> 16) & 3U);
        const int anchor_lane = static_cast<int>((block_key >> 32) & 15U);
        for (std::int32_t lane = 0; lane < kBlock; ++lane) {
            const auto key = mix_fixture_coordinate(block_key + static_cast<std::uint32_t>(lane));
            const std::size_t magnitude = lane == anchor_lane ? 7 : 1 + key % 7;
            const float sign = (key >> 63) != 0 ? -1.0F : 1.0F;
            values[static_cast<std::size_t>(block) * kBlock + lane] =
                f32_to_bf16(std::ldexp(sign * kMagnitudes[magnitude], -exponent));
        }
    }

    // Four distinct active blocks per later token keep the full fused FP64 oracle
    // practical. The coprime stride rotates across all 320 K16 blocks; the selected
    // lane, sign, and exactly represented scale also depend on the token and seed.
    for (std::int32_t token = 1; token < kT; ++token) {
        const auto token_key = mix_fixture_coordinate(
            (static_cast<std::uint64_t>(seed) << 32) | static_cast<std::uint32_t>(token));
        const auto first_block = static_cast<std::int32_t>(token_key % kBlocks);
        for (std::int32_t active = 0; active < 4; ++active) {
            const auto key = mix_fixture_coordinate(token_key + static_cast<std::uint32_t>(active));
            const auto block = (first_block + 73 * active) % kBlocks;
            const auto lane = static_cast<std::int32_t>((key >> 8) & 15U);
            const int exponent = 5 + static_cast<int>((key >> 16) & 3U);
            const float sign = (key >> 63) != 0 ? -1.0F : 1.0F;
            values[static_cast<std::size_t>(token) * kK + block * kBlock + lane] =
                f32_to_bf16(std::ldexp(sign * 6.0F, -exponent));
        }
    }
    return values;
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() {
        (void)cudaStreamSynchronize(value);
        (void)cudaStreamDestroy(value);
    }
};

std::size_t scratch_capacity() {
    return std::max(ops::linear_workspace_capacity_bytes(QType::NVFP4, kN, kK, kPolicy, kT, kT),
                    ops::linear_swiglu_workspace_capacity_bytes(
                        QType::NVFP4, kN, kK, kPolicy, kT, kT));
}

struct Binding {
    linear_swiglu::Profile profile;
    quantized_weight::PackedWeight host_weight;
    std::vector<std::uint16_t> activation;
    std::vector<double> fused_reference;
    std::vector<double> linear_reference;
    GuardedDeviceBuffer device_weight;
    GuardedDeviceBuffer device_activation;
    GuardedDeviceBuffer linear_output;
    GuardedDeviceBuffer fused_output;
    GuardedDeviceBuffer scratch;
    WorkspaceArena workspace;
    Weight weight;

    explicit Binding(std::uint32_t seed)
        : profile{QType::NVFP4, kN, kK, kN / 2, seed, linear_swiglu::ActivationCompute::A4},
          host_weight(linear::make_nvfp4_weight(kN, kK, seed)),
          activation(make_lifetime_activation(seed)),
          fused_reference(linear_swiglu::oracle_fp64(profile, host_weight, activation, kT)),
          linear_reference(kRows.size() * kColumns.size()),
          device_weight(host_weight.payload.size()),
          device_activation(activation.size() * sizeof(std::uint16_t)),
          linear_output(static_cast<std::size_t>(kN) * kT * sizeof(std::uint16_t)),
          fused_output(static_cast<std::size_t>(kN / 2) * kT * sizeof(std::uint16_t)),
          scratch(scratch_capacity()),
          workspace(DeviceSpan{scratch.data(), scratch.bytes()}),
          weight(host_weight.device_weight(device_weight.data())) {
        if (weight.input_scale_divisor != 3.5F) {
            throw std::invalid_argument("lifetime fixture requires input scale divisor 3.5");
        }
        device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
        device_activation.copy_from_host(activation.data(), device_activation.bytes());
        const auto sampled_weight = quantized_weight::materialize_rows_fp32(host_weight, kRows);
        std::vector<float> sampled_activation(static_cast<std::size_t>(kK) * kColumns.size());
        for (std::size_t column = 0; column < kColumns.size(); ++column) {
            for (std::int32_t k = 0; k < kK; ++k) {
                sampled_activation[column * kK + k] =
                    bf16_to_f32(activation[static_cast<std::size_t>(kColumns[column]) * kK + k]);
            }
        }
        linear::cpu_linear_gemm_fp64(sampled_weight.data(), sampled_activation.data(),
                                     linear_reference.data(), static_cast<std::int32_t>(kRows.size()),
                                     kK, static_cast<std::int32_t>(kColumns.size()));
    }

    void enqueue(cudaStream_t stream) {
        cuda_check(cudaMemsetAsync(linear_output.data(), 0xff, linear_output.bytes(), stream),
                   "poison linear output");
        cuda_check(cudaMemsetAsync(fused_output.data(), 0xff, fused_output.bytes(), stream),
                   "poison fused output");
        Tensor input(device_activation.data(), DType::BF16, {kK, kT});
        Tensor projected(linear_output.data(), DType::BF16, {kN, kT});
        Tensor fused(fused_output.data(), DType::BF16, {kN / 2, kT});
        workspace.reset();
        ops::linear(input, weight, projected, kPolicy, workspace, stream);
        ops::linear_swiglu(input, weight, fused, kPolicy, workspace, stream);
    }

    int verify(std::string label) const {
        const auto projected = from_device_bf16(linear_output.data(),
                                                static_cast<std::size_t>(kN) * kT);
        int failures = 0;
        if (!std::all_of(projected.begin(), projected.end(),
                         [](double value) { return std::isfinite(value); })) {
            std::cerr << label << ": linear output contains poison or nonfinite values\n";
            ++failures;
        }
        std::vector<double> sampled;
        sampled.reserve(linear_reference.size());
        for (const std::int32_t column : kColumns) {
            for (const std::int32_t row : kRows) {
                sampled.push_back(projected[static_cast<std::size_t>(column) * kN + row]);
            }
        }
        failures += verify_reduction(label + " Linear", sampled, linear_reference,
                                      ReductionCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
        const auto fused = from_device_bf16(fused_output.data(),
                                            static_cast<std::size_t>(kN / 2) * kT);
        failures += verify_reduction(label + " LinearSwiGLU", fused, fused_reference,
                                      ReductionCriterion{3.3e-3, 5.0e-3, 6.3e-3});
        failures += linear_output.verify_guards(label + " linear output");
        failures += fused_output.verify_guards(label + " fused output");
        failures += device_weight.verify_guards(label + " weight");
        failures += device_activation.verify_guards(label + " activation");
        failures += scratch.verify_guards(label + " workspace");
        return failures;
    }
};

// Captured descriptors must survive this function's stack frame and the graph definition.
// Keeping two executables live also catches accidentally shared mutable descriptor ownership.
struct CapturedGraph {
    cudaStream_t stream;
    cudaGraph_t definition = nullptr;
    cudaGraphExec_t executable = nullptr;

    explicit CapturedGraph(cudaStream_t owner_stream) : stream(owner_stream) {}
    ~CapturedGraph() {
        (void)cudaStreamSynchronize(stream);
        if (executable != nullptr) { (void)cudaGraphExecDestroy(executable); }
        if (definition != nullptr) { (void)cudaGraphDestroy(definition); }
    }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;

    void instantiate() {
        cuda_check(cudaGraphInstantiate(&executable, definition, 0), "instantiate TMA graph");
        cuda_check(cudaGraphDestroy(definition), "destroy TMA graph definition");
        definition = nullptr;
    }

    void launch() const {
        cuda_check(cudaGraphLaunch(executable, stream), "replay TMA graph");
        cuda_check(cudaStreamSynchronize(stream), "retire TMA graph");
    }
};

#if defined(_MSC_VER)
#define NINFER_TEST_NOINLINE __declspec(noinline)
#else
#define NINFER_TEST_NOINLINE __attribute__((noinline))
#endif

NINFER_TEST_NOINLINE void capture_bindings(
    CapturedGraph& graph, const std::array<std::unique_ptr<Binding>, 2>& bindings,
    std::size_t first) {
    cuda_check(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeThreadLocal),
               "begin TMA graph capture");
    try {
        bindings[first]->enqueue(graph.stream);
        bindings[1 - first]->enqueue(graph.stream);
    } catch (...) {
        cudaGraph_t discard = nullptr;
        (void)cudaStreamEndCapture(graph.stream, &discard);
        if (discard != nullptr) { (void)cudaGraphDestroy(discard); }
        throw;
    }
    cuda_check(cudaStreamEndCapture(graph.stream, &graph.definition), "end TMA graph capture");
}

NINFER_TEST_NOINLINE void scrub_host_stack(std::uint32_t seed) {
    // Volatile writes in a separate non-inlined frame overwrite expired launcher stack storage.
    // A captured memcpy must read its owned descriptor, never any of these changing bytes.
    volatile std::uint8_t bytes[128 * 1024];
    for (std::size_t index = 0; index < sizeof(bytes); ++index) {
        bytes[index] = static_cast<std::uint8_t>((index * 37U + seed) & 0xffU);
    }
}

#undef NINFER_TEST_NOINLINE

} // namespace

int main(int argc, char** argv) {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int device = 0;
        if (argc == 3 && std::string(argv[1]) == "--device") {
            device = std::stoi(argv[2]);
        } else if (argc != 1) {
            throw std::invalid_argument("usage: ninfer_nvfp4_tma_descriptors_test [--device N]");
        }
        cuda_check(cudaSetDevice(device), "select device");
        Stream stream;
        // Keep both bindings live: allocator reuse cannot make their four plane addresses equal.
        // Each ordinary/fused launch can reuse the stream-ordered descriptor allocation released
        // by its predecessor. No default-stream synchronization is inserted between consumers.
        std::array<std::unique_ptr<Binding>, 2> bindings{
            std::make_unique<Binding>(1803U), std::make_unique<Binding>(1811U)};
        cuda_check(cudaStreamSynchronize(nullptr), "retire legacy-stream fixture initialization");
        int failures = 0;
        for (int pass = 0; pass < 4; ++pass) {
            for (int call = 0; call < 2; ++call) {
                bindings[static_cast<std::size_t>((pass + call) % 2)]->enqueue(stream.value);
            }
            cuda_check(cudaStreamSynchronize(stream.value), "retire alternating TMA consumers");
            for (std::size_t binding = 0; binding < bindings.size(); ++binding) {
                failures += bindings[binding]->verify("pass " + std::to_string(pass) +
                                                       " binding " + std::to_string(binding));
            }
        }
        std::array<std::unique_ptr<CapturedGraph>, 2> graphs{
            std::make_unique<CapturedGraph>(stream.value),
            std::make_unique<CapturedGraph>(stream.value)};
        for (std::size_t first = 0; first < graphs.size(); ++first) {
            capture_bindings(*graphs[first], bindings, first);
            scrub_host_stack(static_cast<std::uint32_t>(first + 19));
            graphs[first]->instantiate();
        }
        for (int pass = 0; pass < 4; ++pass) {
            for (int call = 0; call < 2; ++call) {
                const std::size_t graph = static_cast<std::size_t>((pass + call) % 2);
                scrub_host_stack(static_cast<std::uint32_t>(pass * 2 + call + 71));
                graphs[graph]->launch();
                // Verify every replay before another graph can overwrite its outputs. Each
                // replay contains its own output poison before both public operator calls.
                for (std::size_t binding = 0; binding < bindings.size(); ++binding) {
                    failures += bindings[binding]->verify(
                        "graph " + std::to_string(graph) + " pass " + std::to_string(pass) +
                        " binding " + std::to_string(binding));
                }
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " eager and captured NVFP4 TMA descriptor reuse on device " << device << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TMA descriptor test: " << error.what() << '\n';
        return 1;
    }
}
