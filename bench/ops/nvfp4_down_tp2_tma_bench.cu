// Private T1024 TP2 down-projection schedule experiment; production dispatch is unchanged.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
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
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;
constexpr int kN = 5120, kK = 8704, kT = 1024;
constexpr std::size_t kElements = static_cast<std::size_t>(kN) * kT;
constexpr std::size_t kScrubBytes = 128ULL << 20;
constexpr ReductionCriterion kCriterion{0.16, 1.0 / 256.0, 0.16};
constexpr std::array<std::uint32_t, 2> kSeeds{1803U, 1811U};
namespace detail = ninfer::ops::detail;
using Geometry = detail::Nvfp4Residual17408Tp2RowGeometry;
using M256S3 = detail::Nvfp4W4a4TmaSchedule<256, 3, 1>;
using M128S3 = detail::Nvfp4W4a4TmaSchedule<128, 3, 1>;
using M128S2 = detail::Nvfp4W4a4TmaSchedule<128, 2, 1>;
static_assert(Geometry::kOutputRows == kN && Geometry::kInputRows == kK);
enum class Route { M256S3, M128S3, M128S2 };
constexpr std::array<Route, 3> kRoutes{Route::M256S3, Route::M128S3, Route::M128S2};
const char* name(Route route) {
    switch (route) {
    case Route::M256S3: return "M256_S3_min1";
    case Route::M128S3: return "M128_S3_min1";
    case Route::M128S2: return "M128_S2_min1";
    }
    throw std::logic_error("unknown route");
}
int token_tile(Route route) { return route == Route::M256S3 ? 256 : 128; }
int stages(Route route) { return route == Route::M128S2 ? 2 : 3; }
struct Options { int device = 0, samples = 31, warmup = 10; bool qualify_only = false; };
Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--qualify-only") { out.qualify_only = true; continue; }
        if (option == "--help" || option == "-h") {
            std::cout << "usage: ninfer_nvfp4_down_tp2_tma_bench [--device N] "
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
struct Oracle {
    std::vector<std::size_t> positions;
    std::vector<double> values;
};
Oracle make_oracle(const qw::PackedWeight& weight,
                   const std::vector<std::uint16_t>& activation,
                   const std::vector<std::uint16_t>& residual, bool add_residual) {
    // Cover every N128 tile boundary and the M128/M256, warp and vector seams.
    // Every sampled output evaluates ALL K8704 products, directly decoding the
    // public weight bytes/scales. No private A4 activation or GPU output is used.
    std::vector<int> rows;
    for (int tile = 0; tile < kN; tile += 128) {
        rows.push_back(tile);
        rows.push_back(tile + 127);
    }
    for (int row : {1, 7, 8, 15, 16, 31, 32, 63, 64, kN - 2}) { rows.push_back(row); }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    constexpr std::array<int, 28> tokens{0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 129,
        255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 769, 895, 896, 1022, 1023};
    Oracle oracle;
    std::vector<double> decoded(kK);
    for (int row : rows) {
        for (int column = 0; column < kK; ++column) {
            decoded[column] = qw::logical_weight_fp64(weight, row, column);
        }
        for (int token : tokens) {
            double sum = 0.0;
            for (int column = 0; column < kK; ++column) {
                sum += decoded[column] * static_cast<double>(
                    bf16_to_f32(activation[static_cast<std::size_t>(token) * kK + column]));
            }
            const std::size_t position = static_cast<std::size_t>(token) * kN + row;
            if (add_residual) { sum += static_cast<double>(bf16_to_f32(residual[position])); }
            oracle.positions.push_back(position);
            oracle.values.push_back(sum);
        }
    }
    return oracle;
}
std::size_t public_workspace_bytes(bool residual) {
    return residual ? ops::linear_add_workspace_capacity_bytes(
                          QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kT, kT)
                    : ops::linear_workspace_capacity_bytes(
                          QType::NVFP4, kN, kK, ops::LinearPolicy::AllowA4, kT, kT);
}
struct Fixture {
    bool add_residual;
    qw::PackedWeight packed;
    std::vector<std::uint16_t> activation, residual;
    Oracle oracle;
    GuardedDeviceBuffer device_weight, device_input, initial_residual, public_output;
    GuardedDeviceBuffer prepared_storage, public_storage;
    WorkspaceArena prepared_arena, public_arena;
    detail::Nvfp4W4a4Workspace prepared;
    Weight weight;
    Tensor input;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 3> outputs;
    // All descriptor bindings stay live through every eager/captured invocation.
    // This isolates scheduling from descriptor allocation/upload overhead.
    std::array<detail::Nvfp4W4a4TmaDescriptors, 3> host_descriptors{};
#ifdef _WIN32
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 3> device_descriptors;
#endif
    std::array<std::size_t, 3> shared_bytes{};
    std::array<int, 3> registers{}, resident_ctas{};
    std::vector<std::uint16_t> baseline;
    std::vector<std::uint8_t> prepared_image;

    Fixture(std::uint32_t seed, bool residual_epilogue, cudaStream_t stream)
        : add_residual(residual_epilogue), packed(make_weight(seed)),
          activation(make_activation(seed)), residual(make_residual(seed)),
          oracle(make_oracle(packed, activation, residual, add_residual)),
          device_weight(packed.payload.size()), device_input(activation.size() * 2),
          initial_residual(kElements * 2), public_output(kElements * 2),
          prepared_storage(detail::nvfp4_w4a4_workspace_capacity_bytes(kT, kK)),
          public_storage(public_workspace_bytes(add_residual)),
          prepared_arena(DeviceSpan{prepared_storage.data(), prepared_storage.bytes()}),
          public_arena(DeviceSpan{public_storage.data(), public_storage.bytes()}),
          prepared(detail::allocate_nvfp4_w4a4_workspace(prepared_arena, kT, kK)),
          weight(packed.device_weight(device_weight.data())),
          input(device_input.data(), DType::BF16, {kK, kT}) {
        device_weight.copy_from_host(packed.payload.data(), packed.payload.size());
        device_input.copy_from_host(activation.data(), device_input.bytes());
        initial_residual.copy_from_host(residual.data(), initial_residual.bytes());
        for (auto& output : outputs) {
            output = std::make_unique<GuardedDeviceBuffer>(kElements * 2);
        }
        // Guard/input copies use the legacy stream; consumers use a nonblocking stream.
        CUDA_CHECK(cudaDeviceSynchronize());
        detail::launch_nvfp4_w4a4_quantize(input, weight, prepared, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        prepared_image.resize(prepared_storage.bytes());
        prepared_storage.copy_to_host(prepared_image.data(), prepared_image.size());
        prepare<M256S3>(Route::M256S3);
        prepare<M128S3>(Route::M128S3);
        prepare<M128S2>(Route::M128S2);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (prepared_arena.used() != prepared_storage.bytes() ||
            prepared_arena.peak_used() != prepared_storage.bytes()) {
            throw std::runtime_error("prepared workspace bound mismatch");
        }
    }
    const char* epilogue_name() const { return add_residual ? "residual" : "identity"; }
    template<class Schedule, class Epilogue>
    void configure(Route route) {
        const auto index = static_cast<std::size_t>(route);
        constexpr std::size_t bytes = sizeof(detail::Nvfp4W4a4TmaSharedStorage<Schedule>);
        const auto kernel = detail::nvfp4_w4a4_tma_kernel<
            Geometry, Schedule, Epilogue, detail::Nvfp4ContiguousOutput>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(bytes)));
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_ctas[index], kernel, Schedule::kThreads, bytes));
        shared_bytes[index] = bytes;
        registers[index] = attributes.numRegs;
        if (resident_ctas[index] < 1) { throw std::runtime_error("schedule is not resident"); }
    }
    template<class Schedule>
    void prepare(Route route) {
        const auto index = static_cast<std::size_t>(route);
        host_descriptors[index] = detail::make_nvfp4_w4a4_tma_descriptors<Geometry, Schedule::kBlockM>(
            prepared.codes, prepared.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), kT);
#ifdef _WIN32
        device_descriptors[index] = std::make_unique<GuardedDeviceBuffer>(sizeof(host_descriptors[index]));
        device_descriptors[index]->copy_from_host(&host_descriptors[index], sizeof(host_descriptors[index]));
#endif
        if (add_residual) { configure<Schedule, detail::Nvfp4AddResidualEpilogue>(route); }
        else { configure<Schedule, detail::Nvfp4IdentityEpilogue>(route); }
    }
    template<class Schedule, class Epilogue>
    void launch_schedule(Route route, Epilogue epilogue, cudaStream_t stream) {
        const auto index = static_cast<std::size_t>(route);
        auto* output = static_cast<__nv_bfloat16*>(outputs[index]->data());
        const dim3 grid(kN / Schedule::kBlockN, kT / Schedule::kBlockM);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        detail::nvfp4_w4a4_tma_kernel<Geometry, Schedule>
            <<<grid, Schedule::kThreads, sizeof(detail::Nvfp4W4a4TmaSharedStorage<Schedule>), stream>>>(
#ifdef _WIN32
                static_cast<const detail::Nvfp4W4a4TmaDescriptors*>(device_descriptors[index]->data()),
#else
                host_descriptors[index],
#endif
                alpha, epilogue, detail::Nvfp4ContiguousOutput{output, kN});
        CUDA_CHECK(cudaGetLastError());
    }
    template<class Schedule>
    void launch_epilogue(Route route, cudaStream_t stream) {
        if (add_residual) {
            auto* output = static_cast<__nv_bfloat16*>(outputs[static_cast<std::size_t>(route)]->data());
            launch_schedule<Schedule>(route, detail::Nvfp4AddResidualEpilogue{output, kN}, stream);
        } else { launch_schedule<Schedule>(route, detail::Nvfp4IdentityEpilogue{}, stream); }
    }
    void launch(Route route, cudaStream_t stream) {
        switch (route) {
        case Route::M256S3: launch_epilogue<M256S3>(route, stream); break;
        case Route::M128S3: launch_epilogue<M128S3>(route, stream); break;
        case Route::M128S2: launch_epilogue<M128S2>(route, stream); break;
        }
    }
    void reset(Route route, cudaStream_t stream) {
        const auto& output = outputs[static_cast<std::size_t>(route)];
        // This copy is outside every event interval. Residual calls always consume
        // the same BF16 input, never the previous invocation's accumulated result.
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
        // Full finite check in addition to the explicitly sampled FP64 criterion.
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (!std::isfinite(bf16_to_f32(bits[i]))) {
                std::cerr << label << ": nonfinite output at " << i << '\n';
                return 1;
            }
        }
        std::vector<double> samples;
        samples.reserve(oracle.positions.size());
        for (const auto position : oracle.positions) { samples.push_back(bf16_to_f32(bits[position])); }
        return verify_reduction(label, samples, oracle.values, kCriterion);
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
            preserved(prepared_storage, prepared_image.data()) + public_storage.verify_guards(phase) +
            public_output.verify_guards(phase);
#ifdef _WIN32
        for (std::size_t i = 0; i < device_descriptors.size(); ++i) {
            failures += preserved(*device_descriptors[i], &host_descriptors[i]);
        }
#endif
        if (public_arena.used() != 0 || public_arena.peak_used() != public_storage.bytes() ||
            prepared_arena.used() != prepared_storage.bytes()) { ++failures; }
        if (failures) { throw std::runtime_error("read-only storage, guards, or arena cursor changed"); }
    }
    void qualify_eager(cudaStream_t stream) {
        Tensor output(public_output.data(), DType::BF16, {kN, kT});
        if (add_residual) {
            CUDA_CHECK(cudaMemcpyAsync(output.data, initial_residual.data(), public_output.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            ops::linear_add(input, weight, output, ops::LinearPolicy::AllowA4, public_arena, stream);
        } else {
            CUDA_CHECK(cudaMemsetAsync(output.data, 0xff, public_output.bytes(), stream));
            ops::linear(input, weight, output, ops::LinearPolicy::AllowA4, public_arena, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        baseline.resize(kElements);
        public_output.copy_to_host(baseline.data(), public_output.bytes());
        // The actual public route must pass independently before serving as an
        // exact comparison. Dense-seed oracle failures abort; no alternate seed.
        if (verify_oracle(baseline, "public production sampled FP64")) {
            throw std::runtime_error("public production fails unchanged A4 criterion");
        }
        for (Route route : kRoutes) { reset(route, stream); launch(route, stream); }
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
    Fixture fixture(seed, residual, stream);
    fixture.qualify_eager(stream);
    DeviceBuffer scrub(kScrubBytes);
    CUDA_CHECK(cudaMemsetAsync(scrub.p, 0, scrub.bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const bool cold : {false, true}) {
        const char* condition = cold ? "scrubbed_128MiB" : "warm";
        std::array<std::unique_ptr<Captured>, 3> graphs;
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
        std::array<std::vector<Measurement>, 3> samples;
        if (!options.qualify_only) {
            // Rotate all three positions; reverse each complete three-round cycle.
            // Every route runs once in each round, against the same baseline round.
            for (int round = -options.warmup; round < options.samples; ++round) {
                const int index = round + options.warmup;
                for (int position = 0; position < 3; ++position) {
                    const int offset = ((index / 3) & 1) ? 2 - position : position;
                    const int route = (index + offset) % 3;
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
                  << "\",\"sampled_complete_k_fp64_passed\":true,\"oracle_outputs\":"
                  << fixture.oracle.positions.size() << ",\"total_outputs\":" << kElements
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
                      << "\",\"token_tile\":" << token_tile(kRoutes[route])
                      << ",\"stages\":" << stages(kRoutes[route])
                      << ",\"ctas\":" << (kN / 128) * (kT / token_tile(kRoutes[route]))
                      << ",\"dynamic_shared_bytes\":" << fixture.shared_bytes[route]
                      << ",\"static_registers\":" << fixture.registers[route]
                      << ",\"occupancy_api_ctas_per_sm\":" << fixture.resident_ctas[route]
                      << ",\"samples\":" << times.size()
                      << ",\"median_us\":" << percentile(times, 0.5)
                      << ",\"p95_us\":" << percentile(times, 0.95)
                      << ",\"median_paired_ratio_to_M256S3\":" << percentile(ratios, 0.5)
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
                  << ",\"rdc\":false,\"kernel_only_timing\":true,\"production_default_changed\":false}\n";
        Stream stream;
        for (const auto seed : kSeeds) {
            for (bool residual : {false, true}) { run_fixture(options, seed, residual, stream.value); }
        }
        std::cout << "{\"kind\":\"result\",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 TP2 down TMA experiment: " << error.what() << '\n';
        return 1;
    }
}
