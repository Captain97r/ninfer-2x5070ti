// A kernel-policy experiment for the 12-query/2-KV-head TP2 shard on sm_120a.
// Uses the production INT8 partial/reduce kernels, changing ONLY a bench-local split cap.
// It deliberately bypasses public dispatch; results are operator timings, not tokens/second.
// The sm70_broad_graph row launches 170 splits but selects 70 active splits from device
// positions, matching the production replay envelope. Every cap is checked against the
// existing independent FP64 oracle over represented BF16 queries and INT8-G64 cache.
// The benchmark does not change production tuning at runtime.
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "ops/gqa_attention_fixture.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::gqa;

namespace {
constexpr fixture::Geometry kGeometry{"d256-h12-kv2", 12, 2};
constexpr std::array<int, 6> kCaps{35, 70, 105, 140, 170, 0};
constexpr std::size_t kFlushBytes = 128u << 20;

// The kernel derives page-ID staging from this same cap, preserving its full-domain
// bound. Merely reducing grid.y with the original 170-split geometry could overrun it.
template <int Cap> struct CandidateGeometry : ops::Gqa27Tp2Geometry {
    static constexpr int DecodeSplits = Cap;
    static constexpr int DecodePageSplitFloor = Cap;
    static constexpr int LongWindowSplits = Cap;
};

struct Options { int device = 0; int context = 100000; int repeat = 31; };
int number(const char* text, int low, int high) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < low || value > high)
        throw std::invalid_argument("integer argument outside the supported range");
    return static_cast<int>(value);
}
Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--help") {
            std::puts("ninfer_gqa_tp2_split_bench [--device N] [--context 40960..1048571] [--repeat 5..1001]\n"
                      "INT8-G64 cached attention, fragmented pages, T=1/4/5, 12Q/2KV, graph replay, cold L2.\n"
                      "Checks every output against FP64 before timing. Caps are bench-local; production is 170.");
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("missing option value");
        if (flag == "--device") options.device = number(argv[i], 0, 31);
        else if (flag == "--context") options.context = number(argv[i], 40960, 1048571);
        else if (flag == "--repeat") options.repeat = number(argv[i], 5, 1001);
        else throw std::invalid_argument("unknown option");
    }
    return options;
}

__global__ void flush_l2(std::uint32_t* buffer, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) buffer[i] = static_cast<std::uint32_t>(i);
}

struct Case {
    PagedKVLayerView cache;
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* partial;
    float* maxima;
    float* sums;
    __nv_bfloat16* output;
    int capacity;
};

template <int Tokens, int Cap> void launch(const Case& c, cudaStream_t stream) {
    using G = std::conditional_t<Cap == 0, ops::Gqa27Tp2Sm70Geometry, CandidateGeometry<Cap>>;
    constexpr int splits = Cap == 0 ? 170 : Cap;
    // The exact production long-window profile for T=1,4,5 and GroupSize=6:
    // 8 warps, two resident CTAs/SM, Bc=32, static shared arena, one unmasked row.
    ops::gqa_attention_decode_i8_tiled_kernel<G, Tokens, 8, 2, 32, false,
                                              false, false, ops::GqaCachedInput>
        <<<dim3(2, splits), 256, 0, stream>>>(
            c.q, {}, c.positions, static_cast<std::int8_t*>(c.cache.k_pages.data),
            static_cast<std::int8_t*>(c.cache.v_pages.data),
            static_cast<__half*>(c.cache.k_scale_pages.data),
            static_cast<__half*>(c.cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(c.cache.block_table.data), nullptr, nullptr,
            c.cache.block_table.ne[0], Tokens, 0, c.capacity, fixture::kAttentionScale,
            c.partial, c.maxima, c.sums);
    ops::gqa_attention_small_t_reduce_output_kernel<G, 64, true, false, false, false>
        <<<dim3(12, 4, Tokens), 256, 0, stream>>>(
            c.partial, c.maxima, c.sums, c.positions, nullptr, Tokens, Tokens, 0, 1,
            splits, c.output);
    cuda_check_last_launch("GQA candidate launch");
}

template <int Tokens> void dispatch(int cap, const Case& c, cudaStream_t stream) {
    switch (cap) {
    case 0: launch<Tokens, 0>(c, stream); break;
    case 35: launch<Tokens, 35>(c, stream); break;
    case 70: launch<Tokens, 70>(c, stream); break;
    case 105: launch<Tokens, 105>(c, stream); break;
    case 140: launch<Tokens, 140>(c, stream); break;
    case 170: launch<Tokens, 170>(c, stream); break;
    default: throw std::logic_error("invalid split cap");
    }
}

struct Graph {
    cudaGraph_t definition = nullptr;
    cudaGraphExec_t executable = nullptr;
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (definition) cudaGraphDestroy(definition);
    }
};
struct TimingResources {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    TimingResources() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        cuda_check(cudaEventCreate(&begin), "create start event");
        cuda_check(cudaEventCreate(&end), "create end event");
    }
    ~TimingResources() {
        if (end) cudaEventDestroy(end);
        if (begin) cudaEventDestroy(begin);
        if (stream) cudaStreamDestroy(stream);
    }
};

template <int Tokens>
void benchmark(const Options& options, const Case& c, DeviceBuffer& output,
               DeviceBuffer& flush, const std::vector<double>& full_reference,
               TimingResources& timing) {
    constexpr std::size_t count = 12 * 256 * Tokens;
    const std::vector<double> reference(full_reference.begin(), full_reference.begin() + count);
    const double floor = fixture::bf16_storage_floor_relative_l2(reference);
    const auto criterion = fixture::long_window_attention_criterion(DType::I8, floor);
    std::array<Graph, kCaps.size()> graphs;
    std::array<std::vector<float>, kCaps.size()> samples;
    std::array<double, kCaps.size()> errors{};
    for (std::size_t i = 0; i < kCaps.size(); ++i) {
        // Check before capture/timing; the FP64 oracle sees no production staging casts.
        output.fill(0x7f);
        dispatch<Tokens>(kCaps[i], c, timing.stream);
        cuda_synchronize(timing.stream);
        const auto actual = from_device_bf16(output, count);
        const auto stats = compute_reduction_stats(actual.data(), reference.data(), count);
        if (!reduction_passes(stats, count, criterion)) {
            verify_reduction("GQA split cap " + std::to_string(kCaps[i]), actual, reference, criterion);
            throw std::runtime_error("GQA candidate failed the independent FP64 oracle");
        }
        errors[i] = stats.relative_l2;
        cuda_check(cudaStreamBeginCapture(timing.stream, cudaStreamCaptureModeThreadLocal), "capture start");
        dispatch<Tokens>(kCaps[i], c, timing.stream);
        cuda_check(cudaStreamEndCapture(timing.stream, &graphs[i].definition), "capture end");
        cuda_check(cudaGraphInstantiate(&graphs[i].executable, graphs[i].definition, 0), "instantiate graph");
        for (int warmup = 0; warmup < 10; ++warmup)
            cuda_check(cudaGraphLaunch(graphs[i].executable, timing.stream), "warmup graph");
        cuda_synchronize(timing.stream);
    }
    // Interleave caps and reverse their order on alternating rounds to reduce clock/thermal bias.
    // Flush outside the event interval, as other layers evict attention's cache in real decode.
    for (int round = 0; round < options.repeat; ++round) {
        for (std::size_t step = 0; step < kCaps.size(); ++step) {
            const std::size_t i = round % 2 ? kCaps.size() - 1 - step : step;
            flush_l2<<<512, 256, 0, timing.stream>>>(
                static_cast<std::uint32_t*>(flush.p), kFlushBytes / sizeof(std::uint32_t));
            cuda_check_last_launch("L2 flush");
            cuda_check(cudaEventRecord(timing.begin, timing.stream), "record start");
            cuda_check(cudaGraphLaunch(graphs[i].executable, timing.stream), "timed graph");
            cuda_check(cudaEventRecord(timing.end, timing.stream), "record end");
            cuda_check(cudaEventSynchronize(timing.end), "wait for timing");
            float elapsed_ms = 0;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, timing.begin, timing.end), "read timing");
            samples[i].push_back(elapsed_ms * 1000);
        }
    }
    for (std::size_t i = 0; i < kCaps.size(); ++i) {
        auto& s = samples[i];
        std::sort(s.begin(), s.end());
        const int splits = kCaps[i] == 0 ? 170 : kCaps[i];
        std::printf("%d,%d,%d,%s,%d,%d,%.3f,%.3f,%.3f,%.6g,%.6g\n", options.device,
                    options.context, Tokens, kCaps[i] == 0 ? "sm70_broad_graph" : "fixed_cap",
                    splits, 2 * splits, s.front(), s[s.size()/2],
                    s[(s.size()-1)*9/10], errors[i], criterion.relative_l2);
    }
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (cuda_unavailable()) return 77;
        cuda_check(cudaSetDevice(options.device), "select device");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, options.device), "read device");
        if (properties.major != 12 || properties.minor != 0)
            throw std::runtime_error("this benchmark requires sm_120a hardware");
        std::fprintf(stderr, "GPU%d: %s, %d SMs; CUDA runtime %d; context=%d; INT8-G64; fragmented pages\n",
                     options.device, properties.name, properties.multiProcessorCount,
                     CUDART_VERSION, options.context);
        auto host_cache = fixture::make_cache(kGeometry, DType::I8, options.context + 5, 5070);
        const auto q = fixture::make_bf16_values(12 * 256 * 5, 17035, -0.25f, 0.25f);
        std::vector<std::int32_t> positions(5);
        for (int t = 0; t < 5; ++t) positions[t] = options.context + t;
        std::fprintf(stderr, "Computing FP64 oracle for all 12 heads and five causal queries...\n");
        const auto reference = fixture::ideal_attention(q, host_cache, positions);
        fixture::DeviceCache cache(host_cache, fixture::MappingPattern::Fragmented);
        auto dq = to_device_bf16(q);
        auto dp = to_device(positions);
        constexpr std::size_t stats_count = 12 * 5 * 170;
        DeviceBuffer partial(stats_count * 256 * sizeof(std::uint16_t));
        DeviceBuffer maxima(stats_count * sizeof(float)), sums(stats_count * sizeof(float));
        DeviceBuffer output(12 * 256 * 5 * sizeof(std::uint16_t));
        DeviceBuffer flush(kFlushBytes);
        TimingResources timing;
        const Case c{cache.view(), static_cast<const __nv_bfloat16*>(dq.p),
                     static_cast<const std::int32_t*>(dp.p),
                     static_cast<__nv_bfloat16*>(partial.p), static_cast<float*>(maxima.p),
                     static_cast<float*>(sums.p), static_cast<__nv_bfloat16*>(output.p),
                     host_cache.logical_capacity};
        std::puts("device,context,tokens,policy,splits,partial_ctas,min_us,median_us,p90_us,relative_l2,l2_limit");
        benchmark<1>(options, c, output, flush, reference, timing);
        benchmark<4>(options, c, output, flush, reference, timing);
        benchmark<5>(options, c, output, flush, reference, timing);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GQA split benchmark failed: %s\n", error.what());
        return 1;
    }
}
