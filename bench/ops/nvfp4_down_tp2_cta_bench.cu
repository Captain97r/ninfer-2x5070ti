// Private T4 TP2 down-projection CTA experiment; production dispatch is unchanged.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;
constexpr int kN = 5120, kK = 8704, kT = 4;
constexpr std::size_t kElements = static_cast<std::size_t>(kN) * kT;
constexpr std::size_t kScrubBytes = 128ULL << 20;
constexpr ReductionCriterion kCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr std::array<std::uint32_t, 2> kSeeds{1803U, 1811U};
namespace detail = ninfer::ops::detail;
using Geometry = detail::Nvfp4Residual17408Tp2RowGeometry;
template<int Warps>
using Schedule = detail::Nvfp4SmallTSchedule<Warps, 1, 2, 16, kT, 1,
    detail::Nvfp4SmallTActivationAccess::TokenPacked, detail::Nvfp4ScaleAccess::Direct,
    detail::Nvfp4CodeCache::Default, 1, detail::Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
static_assert(Geometry::kOutputRows == kN && Geometry::kInputRows == kK);
static_assert(std::is_same_v<Schedule<4>,
    detail::Nvfp4LinearSmallTProductionSchedule<Geometry, kT>::Type>);
enum class Route { Production4, Header4, Header8 };
constexpr std::array kRoutes{Route::Production4, Route::Header4, Route::Header8};
constexpr int kRouteCount = static_cast<int>(kRoutes.size());
const char* name(Route route) {
    switch (route) {
    case Route::Production4: return "production4";
    case Route::Header4: return "header4";
    case Route::Header8: return "header8";
    }
    throw std::logic_error("unknown route");
}
int warps(Route route) { return route == Route::Header8 ? 8 : 4; }
struct Options { int device = 0, samples = 31, warmup = 10; bool qualify_only = false; };
Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--qualify-only") { out.qualify_only = true; continue; }
        if (option == "--help" || option == "-h") {
            std::cout << "usage: ninfer_nvfp4_down_tp2_cta_bench [--device N] "
                         "[--samples N] [--warmup N] [--qualify-only]\n";
            std::exit(0);
        }
        if (++i == argc) { throw std::invalid_argument("missing option value"); }
        const std::string value(argv[i]);
        std::size_t used = 0;
        const int number = std::stoi(value, &used);
        if (used != value.size()) { throw std::invalid_argument("invalid integer option"); }
        if (option == "--device") { out.device = number; }
        else if (option == "--samples") { out.samples = number; }
        else if (option == "--warmup") { out.warmup = number; }
        else { throw std::invalid_argument("unknown option"); }
    }
    if (out.device < 0 || out.samples < 3 || out.warmup < 1) {
        throw std::invalid_argument("device>=0, samples>=3 and warmup>=1 required");
    }
    return out;
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() {
        (void)cudaStreamSynchronize(value);
        (void)cudaStreamDestroy(value);
    }
};
std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
std::vector<std::uint16_t> make_activation(std::uint32_t seed) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(kK) * kT);
    for (int token = 0; token < kT; ++token) {
        for (int column = 0; column < kK; ++column) {
            const auto key = mix((static_cast<std::uint64_t>(seed) << 32) |
                                static_cast<std::uint32_t>(token * kK + column));
            int numerator = static_cast<int>(key % 255U) - 127;
            if (numerator == 0) { numerator = (column & 1) == 0 ? 1 : -1; }
            // EVERY token is dense, mixed-sign, O(1) RMS and represented exactly in BF16.
            values[static_cast<std::size_t>(token) * kK + column] =
                f32_to_bf16(static_cast<float>(numerator) / 64.0F);
        }
    }
    return values;
}
qw::PackedWeight make_weight(std::uint32_t seed) {
    qw::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    options.decorrelate_coordinates = true;
    return qw::make_patterned_weight(QType::NVFP4, kN, kK, seed, options);
}

std::vector<std::uint16_t> make_residual(std::uint32_t seed) {
    std::vector<std::uint16_t> values(kElements);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int numerator = static_cast<int>(mix((static_cast<std::uint64_t>(seed) << 32) ^
                                                   i ^ 0xa53e9127ULL) % 255U) - 127;
        values[i] = f32_to_bf16(static_cast<float>(numerator) / 4.0F);
    }
    return values;
}

std::vector<double> make_oracle(const qw::PackedWeight& weight,
                                const std::vector<std::uint16_t>& activation,
                                const std::vector<std::uint16_t>& residual, bool add_residual) {
    // Complete logical formula for EVERY output. Directly decode signed public
    // weight codes/stored scales; do not reproduce the GPU coefficient rounding,
    // per-lane FMA chains, warp reduction, or final BF16 conversion.
    std::vector<double> oracle(kElements), decoded(kK);
    for (int row = 0; row < kN; ++row) {
        for (int column = 0; column < kK; ++column) {
            decoded[column] = qw::logical_weight_fp64(weight, row, column);
        }
        for (int token = 0; token < kT; ++token) {
            double sum = 0.0;
            for (int column = 0; column < kK; ++column) {
                sum += decoded[column] * static_cast<double>(
                    bf16_to_f32(activation[static_cast<std::size_t>(token) * kK + column]));
            }
            const std::size_t position = static_cast<std::size_t>(token) * kN + row;
            if (add_residual) { sum += static_cast<double>(bf16_to_f32(residual[position])); }
            oracle[position] = sum;
        }
    }
    return oracle;
}
struct Resources {
    int registers = 0, resident_ctas = 0;
    std::size_t shared_bytes = 0, local_bytes = 0;
};
struct Fixture {
    bool add_residual;
    qw::PackedWeight packed;
    std::vector<std::uint16_t> activation, residual;
    std::vector<double> oracle;
    GuardedDeviceBuffer device_weight, device_input, initial_residual, arena_storage;
    WorkspaceArena public_arena;
    Weight weight;
    Tensor input;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, kRouteCount> outputs;
    std::array<Resources, kRouteCount> resources{};
    std::vector<std::uint16_t> baseline;
    std::array<std::uint8_t, 256> unused_arena_image{};

    Fixture(std::uint32_t seed, bool residual_epilogue)
        : add_residual(residual_epilogue), packed(make_weight(seed)),
          activation(make_activation(seed)), residual(make_residual(seed)),
          oracle(make_oracle(packed, activation, residual, add_residual)),
          device_weight(packed.payload.size()), device_input(activation.size() * 2),
          initial_residual(kElements * 2), arena_storage(256),
          public_arena(DeviceSpan{arena_storage.data(), arena_storage.bytes()}),
          weight(packed.device_weight(device_weight.data())),
          input(device_input.data(), DType::BF16, {kK, kT}) {
        const std::size_t workspace = add_residual
            ? ops::linear_add_workspace_capacity_bytes(
                QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kT, kT)
            : ops::linear_workspace_capacity_bytes(
                QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kT, kT);
        if (workspace != 0) { throw std::runtime_error("T4 production changed its A16 workspace route"); }
        unused_arena_image.fill(0x6d);
        arena_storage.copy_from_host(unused_arena_image.data(), unused_arena_image.size());
        device_weight.copy_from_host(packed.payload.data(), packed.payload.size());
        device_input.copy_from_host(activation.data(), device_input.bytes());
        initial_residual.copy_from_host(residual.data(), initial_residual.bytes());
        for (auto& output : outputs) { output = std::make_unique<GuardedDeviceBuffer>(kElements * 2); }
        // Guard/input copies use the legacy stream; consumers use a nonblocking stream.
        CUDA_CHECK(cudaDeviceSynchronize());
        configure<4>(Route::Header4);
        configure<8>(Route::Header8);
    }
    const char* epilogue_name() const { return add_residual ? "residual" : "identity"; }
    template<int Warps, class Epilogue>
    void query_resources(Route route) {
        const auto kernel = detail::nvfp4_small_t_kernel<
            Geometry, kT, Schedule<Warps>, Epilogue, detail::Nvfp4ContiguousOutput>;
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        auto& result = resources[static_cast<std::size_t>(route)];
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &result.resident_ctas, kernel, Schedule<Warps>::kThreads, 0));
        result.registers = attributes.numRegs;
        result.shared_bytes = attributes.sharedSizeBytes;
        result.local_bytes = attributes.localSizeBytes;
        if (result.resident_ctas < 1) { throw std::runtime_error("schedule is not resident"); }
    }
    template<int Warps>
    void configure(Route route) {
        if (add_residual) { query_resources<Warps, detail::Nvfp4AddResidualEpilogue>(route); }
        else { query_resources<Warps, detail::Nvfp4IdentityEpilogue>(route); }
    }
    template<int Warps, class Epilogue>
    void launch_schedule(Route route, Epilogue epilogue, cudaStream_t stream) {
        using Plan = Schedule<Warps>;
        auto* output = static_cast<__nv_bfloat16*>(outputs[static_cast<std::size_t>(route)]->data());
        constexpr int blocks = kN / Plan::kRowsPerCta;
        const float inverse = 1.0F / weight.weight_scale_divisor;
        detail::nvfp4_small_t_kernel<Geometry, kT, Plan>
            <<<blocks, Plan::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), inverse, epilogue,
                detail::Nvfp4ContiguousOutput{output, kN});
        CUDA_CHECK(cudaGetLastError());
    }
    template<int Warps>
    void launch_epilogue(Route route, cudaStream_t stream) {
        if (add_residual) {
            auto* output = static_cast<__nv_bfloat16*>(outputs[static_cast<std::size_t>(route)]->data());
            launch_schedule<Warps>(route, detail::Nvfp4AddResidualEpilogue{output, kN}, stream);
        } else { launch_schedule<Warps>(route, detail::Nvfp4IdentityEpilogue{}, stream); }
    }
    void launch(Route route, cudaStream_t stream) {
        if (route == Route::Production4) {
            Tensor output(outputs[0]->data(), DType::BF16, {kN, kT});
            // Use the shipping permissive policy, which resolves this T4 shape to
            // A16. Both public operators admit the actual TP2 row shard directly.
            if (add_residual) {
                ops::linear_add(input, weight, output, ops::LinearPolicy::AllowA4, public_arena, stream);
            } else { ops::linear(input, weight, output, ops::LinearPolicy::AllowA4, public_arena, stream); }
        } else if (route == Route::Header4) { launch_epilogue<4>(route, stream); }
        else { launch_epilogue<8>(route, stream); }
    }
    void reset(Route route, cudaStream_t stream) {
        const auto& output = outputs[static_cast<std::size_t>(route)];
        // Excluded from the event interval on EVERY route. The residual formula is
        // BF16(FP32 row sum + FP32(BF16 residual)); previous outputs are never reused.
        if (add_residual) {
            CUDA_CHECK(cudaMemcpyAsync(output->data(), initial_residual.data(), output->bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        } else { CUDA_CHECK(cudaMemsetAsync(output->data(), 0xff, output->bytes(), stream)); }
    }
    std::vector<std::uint16_t> read(Route route) const {
        std::vector<std::uint16_t> bits(kElements);
        outputs[static_cast<std::size_t>(route)]->copy_to_host(bits.data(), bits.size() * 2);
        return bits;
    }
    int verify_oracle(const std::vector<std::uint16_t>& bits, std::string_view label) const {
        std::vector<double> values(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
        return verify_reduction(label, values, oracle, kCriterion);
    }
    void verify(Route route, std::string_view phase) const {
        const auto bits = read(route);
        const std::string label = std::string(phase) + " " + epilogue_name() + " " + name(route);
        const int failures = verify_oracle(bits, label) +
            verify_exact(label.c_str(), bits, baseline) +
            outputs[static_cast<std::size_t>(route)]->verify_guards(label);
        if (failures) { throw std::runtime_error(label + " qualification failed"); }
    }
    void verify_inputs(std::string_view phase) const {
        const auto preserved = [&](const GuardedDeviceBuffer& buffer, const void* expected) {
            std::vector<std::uint8_t> actual(buffer.bytes());
            buffer.copy_to_host(actual.data(), actual.size());
            const auto* bytes = static_cast<const std::uint8_t*>(expected);
            return (std::equal(actual.begin(), actual.end(), bytes) ? 0 : 1) +
                   buffer.verify_guards(phase);
        };
        int failures = preserved(device_weight, packed.payload.data()) +
            preserved(device_input, activation.data()) + preserved(initial_residual, residual.data()) +
            preserved(arena_storage, unused_arena_image.data());
        if (public_arena.used() != 0 || public_arena.peak_used() != 0) { ++failures; }
        if (failures) { throw std::runtime_error("read-only storage, guards, or zero-workspace contract changed"); }
    }
    void qualify_eager(cudaStream_t stream) {
        reset(Route::Production4, stream);
        launch(Route::Production4, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        baseline = read(Route::Production4);
        // Qualify the actual public route independently before using its BF16 bits.
        if (verify_oracle(baseline, "public production complete FP64")) {
            throw std::runtime_error("public production fails unchanged A16 criterion");
        }
        for (Route route : {Route::Header4, Route::Header8}) { reset(route, stream); launch(route, stream); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (Route route : kRoutes) { verify(route, "eager"); }
        verify_inputs("eager");
    }
};
// Read/write the entire buffer outside the event interval. This perturbs caches
// instead of repeatedly timing one resident weight. No DRAM-bandwidth claim follows.
__global__ void scrub_cache(std::uint32_t* values, std::size_t words) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        values[i] = values[i] * 1664525U + 1013904223U;
    }
}
struct Measurement { double kernel_us, enqueue_us, whole_graph_wall_us; };
struct Captured {
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    Captured(Fixture& fixture, Route route, DeviceBuffer* scrub, cudaStream_t owner)
        : stream(owner) {
        try {
            CUDA_CHECK(cudaEventCreate(&begin));
            CUDA_CHECK(cudaEventCreate(&end));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try {
                fixture.reset(route, stream);
                if (scrub) {
                    scrub_cache<<<1024, 256, 0, stream>>>(
                        static_cast<std::uint32_t*>(scrub->p), scrub->bytes / 4);
                    CUDA_CHECK(cudaGetLastError());
                }
                // External event nodes record on every replay; default captured events
                // may only form dependencies and cannot supply replay timing.
                CUDA_CHECK(cudaEventRecordWithFlags(begin, stream, cudaEventRecordExternal));
                fixture.launch(route, stream);
                CUDA_CHECK(cudaEventRecordWithFlags(end, stream, cudaEventRecordExternal));
            } catch (...) {
                cudaGraph_t discard = nullptr;
                (void)cudaStreamEndCapture(stream, &discard);
                if (discard) { (void)cudaGraphDestroy(discard); }
                throw;
            }
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
            CUDA_CHECK(cudaGraphUpload(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) { release(); throw; }
    }
    ~Captured() { release(); }
    Captured(const Captured&) = delete;
    Captured& operator=(const Captured&) = delete;
    void release() noexcept {
        (void)cudaStreamSynchronize(stream);
        if (executable) { (void)cudaGraphExecDestroy(executable); executable = nullptr; }
        if (graph) { (void)cudaGraphDestroy(graph); graph = nullptr; }
        if (end) { (void)cudaEventDestroy(end); end = nullptr; }
        if (begin) { (void)cudaEventDestroy(begin); begin = nullptr; }
    }
    Measurement run() const {
        const auto first = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        const auto enqueued = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto done = std::chrono::steady_clock::now();
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        if (!std::isfinite(ms) || ms <= 0.0F) {
            throw std::runtime_error("invalid captured event timing");
        }
        return {static_cast<double>(ms) * 1000.0,
                std::chrono::duration<double, std::micro>(enqueued - first).count(),
                std::chrono::duration<double, std::micro>(done - first).count()};
    }
};
double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(fraction * values.size()) - 1)];
}
void run_fixture(const Options& options, std::uint32_t seed, bool residual, cudaStream_t stream) {
    Fixture fixture(seed, residual);
    fixture.qualify_eager(stream);
    // These attributes belong to the header instantiations. Public library
    // code generation is checked by output identity and its own timing route.
    for (Route route : {Route::Header4, Route::Header8}) {
        const auto& resource = fixture.resources[static_cast<std::size_t>(route)];
        std::cout << "{\"kind\":\"kernel_resources\",\"device\":" << options.device
                  << ",\"seed\":" << seed << ",\"epilogue\":\"" << fixture.epilogue_name()
                  << "\",\"route\":\"" << name(route) << "\",\"static_registers\":" << resource.registers
                  << ",\"static_shared_bytes\":" << resource.shared_bytes
                  << ",\"local_bytes_per_thread\":" << resource.local_bytes
                  << ",\"occupancy_api_ctas_per_sm\":" << resource.resident_ctas << "}\n";
    }
    DeviceBuffer scrub(kScrubBytes);
    CUDA_CHECK(cudaMemsetAsync(scrub.p, 0, scrub.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const bool cold : {false, true}) {
        const char* condition = cold ? "scrubbed_128MiB" : "warm";
        std::array<std::unique_ptr<Captured>, kRouteCount> graphs;
        for (Route route : kRoutes) {
            graphs[static_cast<std::size_t>(route)] =
                std::make_unique<Captured>(fixture, route, cold ? &scrub : nullptr, stream);
        }
        const auto qualify = [&](const char* phase) {
            for (Route route : kRoutes) {
                (void)graphs[static_cast<std::size_t>(route)]->run();
                fixture.verify(route, phase);
            }
            fixture.verify_inputs(phase);
        };
        qualify("captured-before");
        std::array<std::vector<Measurement>, kRouteCount> samples;
        if (!options.qualify_only) {
            // Rotate all three positions; reverse each complete three-round cycle.
            // Every route runs once in each round, against the same baseline round.
            for (int round = -options.warmup; round < options.samples; ++round) {
                const int index = round + options.warmup;
                for (int position = 0; position < kRouteCount; ++position) {
                    const int offset = ((index / kRouteCount) & 1)
                                           ? kRouteCount - 1 - position : position;
                    const int route = (index + offset) % kRouteCount;
                    const Measurement sample = graphs[static_cast<std::size_t>(route)]->run();
                    if (round >= 0) {
                        samples[static_cast<std::size_t>(route)].push_back(sample);
                        std::cout << "{\"kind\":\"sample\",\"device\":" << options.device
                                  << ",\"seed\":" << seed << ",\"epilogue\":\"" << fixture.epilogue_name()
                                  << "\",\"cache\":\"" << condition
                                  << "\",\"pair\":" << round << ",\"order\":" << position
                                  << ",\"route\":\"" << name(kRoutes[static_cast<std::size_t>(route)])
                                  << "\",\"kernel_us\":" << sample.kernel_us
                                  << ",\"enqueue_us\":" << sample.enqueue_us
                                  << ",\"whole_graph_wall_us\":" << sample.whole_graph_wall_us << "}\n";
                    }
                }
            }
        }
        if (!options.qualify_only) {
            // Inspect the outputs actually left by timing before a fresh replay can
            // overwrite a timing-only failure. Every route owns a distinct output.
            for (Route route : kRoutes) { fixture.verify(route, "timed-final"); }
            fixture.verify_inputs("timed-final");
        }
        qualify("captured-after");
        std::cout << "{\"kind\":\"qualification\",\"device\":" << options.device
                  << ",\"seed\":" << seed << ",\"epilogue\":\"" << fixture.epilogue_name()
                                  << "\",\"cache\":\"" << condition
                  << "\",\"all_outputs_complete_k_fp64_passed\":true,\"oracle_outputs\":"
                  << fixture.oracle.size() << ",\"total_outputs\":" << kElements
                  << ",\"all_outputs_production_bits_equal\":true,"
                     "\"guards_and_inputs_unchanged\":true}\n";
        if (options.qualify_only) { continue; }
        for (std::size_t route = 0; route < graphs.size(); ++route) {
            std::vector<double> times, ratios;
            for (std::size_t pair = 0; pair < samples[route].size(); ++pair) {
                times.push_back(samples[route][pair].kernel_us);
                ratios.push_back(samples[route][pair].kernel_us / samples[0][pair].kernel_us);
            }
            std::cout << "{\"kind\":\"summary\",\"device\":" << options.device
                      << ",\"seed\":" << seed << ",\"epilogue\":\"" << fixture.epilogue_name()
                                  << "\",\"cache\":\"" << condition
                      << "\",\"route\":\"" << name(kRoutes[route])
                      << "\",\"warps_per_cta\":" << warps(kRoutes[route])
                      << ",\"ctas\":" << kN / (2 * warps(kRoutes[route]))
                      << ",\"samples\":" << times.size()
                      << ",\"median_us\":" << percentile(times, 0.5)
                      << ",\"p95_us\":" << percentile(times, 0.95)
                      << ",\"median_paired_ratio_to_production4\":" << percentile(ratios, 0.5)
                      << "}\n";
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (ninfer::test::cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
        std::cout << std::setprecision(12)
                  << "{\"kind\":\"config\",\"device\":" << options.device
                  << ",\"gpu\":" << std::quoted(properties.name)
                  << ",\"sm_count\":" << properties.multiProcessorCount
                  << ",\"N\":" << kN << ",\"K\":" << kK << ",\"T\":" << kT
                  << ",\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
                  << ",\"scrub_bytes\":" << kScrubBytes
                  << ",\"oracle_relative_l2\":" << kCriterion.relative_l2
                  << ",\"oracle_gross_absolute\":" << kCriterion.gross_absolute
                  << ",\"oracle_gross_relative\":" << kCriterion.gross_relative_to_max_reference
                  << ",\"rdc\":true,\"kernel_only_timing\":true,\"production_default_changed\":false}\n";
        Stream stream;
        for (const auto seed : kSeeds) {
            for (bool residual : {false, true}) { run_fixture(options, seed, residual, stream.value); }
        }
        std::cout << "{\"kind\":\"result\",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TP2 down T4 CTA experiment: " << error.what() << '\n';
        return 1;
    }
}
