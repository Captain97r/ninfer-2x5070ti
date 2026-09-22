// Isolated TP2 [17408,5120], T4 A16 experiment. Public single-device admission
// remains [34816,5120]; the baseline calls the actual private shard dispatcher.
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cuh"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
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
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;
constexpr int kN = 17408, kK = 5120, kT = 4, kM = kN / 2;
constexpr std::size_t kElements = static_cast<std::size_t>(kM) * kT;
constexpr std::size_t kScrubBytes = 128ULL << 20;
constexpr ReductionCriterion kCriterion{3.3e-3, 5.0e-3, 6.3e-3};
constexpr std::array<std::uint32_t, 2> kSeeds{1803U, 1811U};
enum class Route { Production8, Header8, Header4, Header16 };
constexpr std::array<Route, 4> kRoutes{
    Route::Production8, Route::Header8, Route::Header4, Route::Header16};
const char* name(Route route) {
    switch (route) {
    case Route::Production8: return "production8";
    case Route::Header8: return "header8";
    case Route::Header4: return "header4";
    case Route::Header16: return "header16";
    }
    throw std::logic_error("unknown route");
}
int warps(Route route) {
    return route == Route::Header4 ? 4 : (route == Route::Header16 ? 16 : 8);
}
struct Options { int device = 0, samples = 31, warmup = 10; bool qualify_only = false; };
Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--qualify-only") { out.qualify_only = true; continue; }
        if (option == "--help" || option == "-h") {
            std::cout << "usage: ninfer_nvfp4_swiglu_tp2_cta_bench [--device N] "
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
struct Fixture {
    qw::PackedWeight packed;
    std::vector<std::uint16_t> activation;
    std::vector<double> oracle;
    GuardedDeviceBuffer device_weight, device_input;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 4> outputs;
    Weight weight;
    Tensor input;
    std::vector<std::uint16_t> baseline;

    explicit Fixture(std::uint32_t seed)
        : packed(make_weight(seed)), activation(make_activation(seed)),
          oracle(linear_swiglu::oracle_fp64(
              {QType::NVFP4, kN, kK, kM, seed, linear_swiglu::ActivationCompute::A16},
              packed, activation, kT)),
          device_weight(packed.payload.size()), device_input(activation.size() * 2),
          weight(packed.device_weight(device_weight.data())),
          input(device_input.data(), DType::BF16, {kK, kT}) {
        device_weight.copy_from_host(packed.payload.data(), packed.payload.size());
        device_input.copy_from_host(activation.data(), device_input.bytes());
        for (auto& output : outputs) {
            output = std::make_unique<GuardedDeviceBuffer>(kElements * 2);
        }
        // Copies and guards use the legacy stream; consumers use a nonblocking stream.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void launch(Route route, cudaStream_t stream) {
        Tensor output(outputs[static_cast<std::size_t>(route)]->data(), DType::BF16, {kM, kT});
        using Geometry = ops::detail::Nvfp4MlpGateUpTp2ColumnGeometry;
        switch (route) {
        case Route::Production8:
            ops::detail::nvfp4_linear_swiglu_small_t_launch_shard(input, weight, output, stream);
            break;
        case Route::Header8:
            ops::detail::launch_nvfp4_swiglu_small_t<Geometry, kT, 8>(input, weight, output, stream);
            break;
        case Route::Header4:
            ops::detail::launch_nvfp4_swiglu_small_t<Geometry, kT, 4>(input, weight, output, stream);
            break;
        case Route::Header16:
            ops::detail::launch_nvfp4_swiglu_small_t<Geometry, kT, 16>(input, weight, output, stream);
            break;
        }
    }
    void poison(Route route, cudaStream_t stream) {
        const auto& output = outputs[static_cast<std::size_t>(route)];
        CUDA_CHECK(cudaMemsetAsync(output->data(), 0xff, output->bytes(), stream));
    }
    std::vector<std::uint16_t> read(Route route) const {
        std::vector<std::uint16_t> bits(kElements);
        outputs[static_cast<std::size_t>(route)]->copy_to_host(bits.data(), bits.size() * 2);
        return bits;
    }
    void verify(Route route, std::string_view phase) const {
        const auto bits = read(route);
        std::vector<double> values(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = bf16_to_f32(bits[i]); }
        const std::string label = std::string(phase) + " " + name(route);
        const int failures = verify_reduction(label, values, oracle, kCriterion) +
            verify_exact(label.c_str(), bits, baseline) +
            outputs[static_cast<std::size_t>(route)]->verify_guards(label);
        if (failures) { throw std::runtime_error(label + " qualification failed"); }
    }
    void verify_inputs(std::string_view phase) const {
        std::vector<std::uint8_t> actual_weight(packed.payload.size());
        device_weight.copy_to_host(actual_weight.data(), actual_weight.size());
        std::vector<std::uint16_t> actual_input(activation.size());
        device_input.copy_to_host(actual_input.data(), device_input.bytes());
        const int failures = verify_exact("weight unchanged", actual_weight, packed.payload) +
            verify_exact("activation unchanged", actual_input, activation) +
            device_weight.verify_guards(phase) + device_input.verify_guards(phase);
        if (failures) { throw std::runtime_error("read-only fixture storage changed"); }
    }
    void qualify_eager(cudaStream_t stream) {
        for (Route route : kRoutes) { poison(route, stream); launch(route, stream); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        baseline = read(Route::Production8);
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
void run_fixture(const Options& options, std::uint32_t seed, cudaStream_t stream) {
    Fixture fixture(seed);
    fixture.qualify_eager(stream);
    DeviceBuffer scrub(kScrubBytes);
    CUDA_CHECK(cudaMemsetAsync(scrub.p, 0, scrub.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const bool cold : {false, true}) {
        const char* condition = cold ? "scrubbed_128MiB" : "warm";
        std::array<std::unique_ptr<Captured>, 4> graphs;
        for (Route route : kRoutes) {
            graphs[static_cast<std::size_t>(route)] =
                std::make_unique<Captured>(fixture, route, cold ? &scrub : nullptr, stream);
        }
        const auto qualify = [&](const char* phase) {
            for (Route route : kRoutes) {
                fixture.poison(route, stream);
                (void)graphs[static_cast<std::size_t>(route)]->run();
                fixture.verify(route, phase);
            }
            fixture.verify_inputs(phase);
        };
        qualify("captured-before");
        std::array<std::vector<Measurement>, 4> samples;
        if (!options.qualify_only) {
            // Rotate all four positions; reverse each complete four-round cycle.
            // Every route runs once in each round, against the same baseline round.
            for (int round = -options.warmup; round < options.samples; ++round) {
                const int index = round + options.warmup;
                for (int position = 0; position < 4; ++position) {
                    const int offset = ((index / 4) & 1) ? 3 - position : position;
                    const int route = (index + offset) % 4;
                    const Measurement sample = graphs[static_cast<std::size_t>(route)]->run();
                    if (round >= 0) {
                        samples[static_cast<std::size_t>(route)].push_back(sample);
                        std::cout << "{\"kind\":\"sample\",\"device\":" << options.device
                                  << ",\"seed\":" << seed << ",\"cache\":\"" << condition
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
                  << ",\"seed\":" << seed << ",\"cache\":\"" << condition
                  << "\",\"all_outputs_fp64_passed\":true,\"all_outputs_baseline_bits_equal\":true,"
                     "\"guards_and_inputs_unchanged\":true}\n";
        if (options.qualify_only) { continue; }
        for (std::size_t route = 0; route < graphs.size(); ++route) {
            std::vector<double> times, ratios;
            for (std::size_t pair = 0; pair < samples[route].size(); ++pair) {
                times.push_back(samples[route][pair].kernel_us);
                ratios.push_back(samples[route][pair].kernel_us / samples[0][pair].kernel_us);
            }
            std::cout << "{\"kind\":\"summary\",\"device\":" << options.device
                      << ",\"seed\":" << seed << ",\"cache\":\"" << condition
                      << "\",\"route\":\"" << name(kRoutes[route])
                      << "\",\"warps_per_cta\":" << warps(kRoutes[route])
                      << ",\"ctas\":" << kM / warps(kRoutes[route])
                      << ",\"samples\":" << times.size()
                      << ",\"median_us\":" << percentile(times, 0.5)
                      << ",\"p95_us\":" << percentile(times, 0.95)
                      << ",\"median_paired_ratio_to_production8\":" << percentile(ratios, 0.5)
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
                  << ",\"production_default_changed\":false}\n";
        Stream stream;
        for (const auto seed : kSeeds) { run_fixture(options, seed, stream.value); }
        std::cout << "{\"kind\":\"result\",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TP2 CTA experiment: " << error.what() << '\n';
        return 1;
    }
}
