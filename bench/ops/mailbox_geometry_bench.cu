// Isolated captured TP2 mailbox geometry experiment. Production dispatch is unchanged.
// Both routes execute the SAME production kernel, in place, with the same protocol.
// Graphs contain 137 homogeneous exchanges (optionally producer-skew kernels), not a
// complete model round. Restoring inputs and resetting retired flags are outside timing.
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/kernel/peer_exchange.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kCalls = 137;
constexpr std::size_t kGuard = 256;
constexpr std::uint8_t kCanary = 0xa5;
constexpr unsigned long long kSkewCycles = 32768;
constexpr std::array<const char*, 2> kRouteNames{"production-grid", "one-block"};

struct Options {
    std::array<int, 2> devices{0, 1};
    int samples = 31, warmup = 5;
    bool qualify_only = false;
};
Options parse(int argc, char** argv) {
    Options options;
    const auto integer = [&](int& index) {
        if (++index >= argc) { throw std::invalid_argument("missing option value"); }
        std::size_t used = 0;
        const std::string text(argv[index]);
        const int result = std::stoi(text, &used);
        if (used != text.size()) { throw std::invalid_argument("invalid integer option"); }
        return result;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--qualify-only") { options.qualify_only = true; }
        else if (argument == "--devices") {
            options.devices[0] = integer(i);
            options.devices[1] = integer(i);
        } else if (argument == "--samples") { options.samples = integer(i); }
        else if (argument == "--warmup") { options.warmup = integer(i); }
        else if (argument == "--help" || argument == "-h") {
            std::cout << "usage: ninfer_mailbox_geometry_bench [--devices A B] "
                         "[--samples N] [--warmup N] [--qualify-only]\n";
            std::exit(0);
        } else { throw std::invalid_argument("unknown option"); }
    }
    if (options.devices[0] < 0 || options.devices[1] < 0 ||
        options.devices[0] == options.devices[1] || options.samples < 3 || options.warmup < 1) {
        throw std::invalid_argument("distinct nonnegative devices, samples>=3, warmup>=1 required");
    }
    return options;
}

// Test-only storage: the retained device id also makes unwinding safe after a CUDA failure.
struct DeviceBytes {
    int device;
    std::size_t size;
    void* data = nullptr;
    DeviceBytes(int selected, std::size_t bytes) : device(selected), size(bytes) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&data, size));
    }
    ~DeviceBytes() {
        (void)cudaSetDevice(device);
        (void)cudaFree(data);
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    std::vector<std::uint8_t> read() const {
        CUDA_CHECK(cudaSetDevice(device));
        std::vector<std::uint8_t> result(size);
        CUDA_CHECK(cudaMemcpy(result.data(), data, size, cudaMemcpyDeviceToHost));
        return result;
    }
};

// One owner per candidate graph family. Payload slots, flag arrays and the fault word each
// have independent canaries; flag/fault addresses retain production's mapped-host protocol.
// No process-global mailbox installation, staged copy, or route-selection fallback is involved.
struct HostMailbox {
    std::size_t payload_bytes, stride, payload_span, flag_span, hang_offset, total;
    std::uint8_t* base = nullptr;
    explicit HostMailbox(std::size_t bytes)
        : payload_bytes(bytes), stride(bytes + 2 * kGuard), payload_span(kCalls * stride),
          flag_span(((kCalls * sizeof(std::uint32_t) + 255) / 256) * 256 + 2 * kGuard),
          hang_offset(2 * payload_span + 2 * flag_span), total(hang_offset + 3 * kGuard) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&base), total, cudaHostAllocMapped));
        std::memset(base, kCanary, total);
        reset(true);
    }
    ~HostMailbox() { (void)cudaFreeHost(base); }
    HostMailbox(const HostMailbox&) = delete;
    HostMailbox& operator=(const HostMailbox&) = delete;
    std::uint8_t* payload(int rank, int call) const {
        return base + rank * payload_span + call * stride + kGuard;
    }
    volatile std::uint32_t* flag(int rank, int call) const {
        return reinterpret_cast<volatile std::uint32_t*>(
            base + 2 * payload_span + rank * flag_span + kGuard) + call;
    }
    volatile std::uint32_t* hang() const {
        return reinterpret_cast<volatile std::uint32_t*>(base + hang_offset + kGuard);
    }
    void reset(bool poison_payload) {
        for (int rank = 0; rank < 2; ++rank) {
            for (int call = 0; call < kCalls; ++call) {
                *flag(rank, call) = 0;
                if (poison_payload) { std::memset(payload(rank, call), 0xcd, payload_bytes); }
            }
        }
        *hang() = 0;
    }
};

void require_bytes(const std::uint8_t* actual, const std::uint8_t* expected,
                   std::size_t bytes, const std::string& label) {
    if (std::memcmp(actual, expected, bytes) == 0) { return; }
    std::size_t at = 0;
    while (at < bytes && actual[at] == expected[at]) { ++at; }
    throw std::runtime_error(label + " differs at byte " + std::to_string(at) +
        ": actual=" + std::to_string(actual[at]) + " expected=" + std::to_string(expected[at]));
}
void require_canary(const std::uint8_t* data, std::size_t bytes, const std::string& label) {
    for (std::size_t i = 0; i < bytes; ++i) {
        if (data[i] != kCanary) {
            throw std::runtime_error(label + " guard differs at byte " + std::to_string(i));
        }
    }
}

std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
// Independent represented-value oracle. Decode sign/exponent/fraction directly in FP64;
// round the mathematical sum directly to BF16 with integer ties-to-even. No CUDA BF16
// conversion, production combine, or FP32 staging is reused by the oracle.
double decode_bf16(std::uint16_t bits) {
    const int exponent = (bits >> 7) & 255;
    const int fraction = bits & 127;
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -133)
        : std::ldexp(static_cast<double>(128 + fraction), exponent - 134);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}
unsigned round_even(double value) {
    const double floor_value = std::floor(value);
    const double remainder = value - floor_value;
    unsigned integer = static_cast<unsigned>(floor_value);
    if (remainder > 0.5 || (remainder == 0.5 && (integer & 1U) != 0)) { ++integer; }
    return integer;
}
std::uint16_t encode_bf16(double value) {
    const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) { return sign; }
    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent);
    int stored_exponent = exponent + 126;
    if (stored_exponent <= 0) {
        return static_cast<std::uint16_t>(sign | round_even(std::ldexp(magnitude, 133)));
    }
    unsigned significand = round_even(fraction * 256.0);
    if (significand == 256) { significand = 128; ++stored_exponent; }
    if (stored_exponent >= 255) { return static_cast<std::uint16_t>(sign | 0x7f80); }
    return static_cast<std::uint16_t>(sign | (stored_exponent << 7) | (significand - 128));
}
std::array<std::uint16_t, 2> operands(int epoch, int call, int element) {
    constexpr std::array<std::array<std::uint16_t, 2>, 10> edge{{
        {{0x0000, 0x8000}}, {{0x8000, 0x8000}}, {{0x3f80, 0x3b80}},
        {{0x3f81, 0x3b80}}, {{0x3f80, 0xbf80}}, {{0x4980, 0x3300}},
        {{0x3f81, 0xbf80}}, {{0xbf81, 0x3f80}}, {{0x7f7f, 0x7f7f}},
        {{0xff7f, 0xff7f}}
    }};
    const auto key = mix((static_cast<std::uint64_t>(epoch + 1) << 48) ^
                         (static_cast<std::uint64_t>(call + 1) << 32) ^
                         static_cast<unsigned>(element));
    const int pattern = (element + call + epoch) % 16;
    std::array<std::uint16_t, 2> result;
    if (pattern < static_cast<int>(edge.size())) { result = edge[pattern]; }
    else {
        for (int rank = 0; rank < 2; ++rank) {
            const auto bits = mix(key + rank);
            result[rank] = static_cast<std::uint16_t>(((bits >> 25) & 0x8000) |
                ((110 + (bits % 31)) << 7) | ((bits >> 16) & 127));
        }
    }
    if ((call + epoch) % 2 != 0) { std::swap(result[0], result[1]); }
    return result;
}

__global__ void producer_skew(unsigned long long cycles) {
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) { __nanosleep(100); }
}

struct RankStorage {
    DeviceBytes seed;
    std::array<std::unique_ptr<DeviceBytes>, 2> work, arrivals;
    RankStorage(int device, std::size_t bytes) : seed(device, bytes) {
        for (int route = 0; route < 2; ++route) {
            work[route] = std::make_unique<DeviceBytes>(device, bytes);
            arrivals[route] = std::make_unique<DeviceBytes>(
                device, kCalls * sizeof(std::uint32_t) + 2 * kGuard);
            CUDA_CHECK(cudaMemset(arrivals[route]->data, kCanary, arrivals[route]->size));
            CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(arrivals[route]->data) + kGuard,
                                  0, kCalls * sizeof(std::uint32_t)));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};
struct GraphRoute {
    HostMailbox mailbox;
    DecodeGraphPeerBridge bridge;
    std::array<DecodeGraphDefinition, 2> definition;
    std::array<DecodeGraphExecutable, 2> executable;
    GraphRoute(const std::array<int, 2>& devices, std::size_t bytes)
        : mailbox(bytes), bridge(devices[0], devices[1]) {}
};

struct Sample { double enqueue_us, wall_us, joined_us; };
struct Fixture {
    ExecutionContext execution;
    int bytes, elements;
    std::size_t stride, image_bytes;
    std::array<std::vector<std::uint8_t>, 2> ingress;
    std::vector<std::uint8_t> expected;
    std::array<std::unique_ptr<RankStorage>, 2> ranks;
    std::array<std::unique_ptr<GraphRoute>, 2> routes;
    cudaEvent_t begin = nullptr, end = nullptr;

    Fixture(const std::array<int, 2>& devices, int payload_bytes)
        : execution({devices[0], devices[1]}), bytes(payload_bytes), elements(bytes / 2),
          stride(bytes + 2 * kGuard), image_bytes(kCalls * stride) {
        for (int rank = 0; rank < 2; ++rank) {
            ranks[rank] = std::make_unique<RankStorage>(devices[rank], image_bytes);
            ingress[rank].resize(image_bytes);
        }
        expected.resize(image_bytes);
        for (int route = 0; route < 2; ++route) {
            select(0);
            routes[route] = std::make_unique<GraphRoute>(devices, bytes);
            for (int skew = 0; skew < 2; ++skew) { capture(route, skew != 0); }
        }
        select(0);
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
    }
    ~Fixture() {
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            (void)cudaStreamSynchronize(execution.dev[rank]->stream);
        }
        // Graphs retire and are destroyed before either captured host/device allocation.
        for (auto& route : routes) { route.reset(); }
        for (auto& rank : ranks) { rank.reset(); }
        (void)cudaSetDevice(execution.dev[0]->device);
        (void)cudaEventDestroy(begin);
        (void)cudaEventDestroy(end);
    }
    void select(int rank) const { CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device)); }
    void retire() const {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaStreamSynchronize(execution.dev[rank]->stream));
        }
    }
    int blocks(int route) const {
        return route == 0 ? ops::detail::peer_exchange_blocks(bytes) : 1;
    }
    void capture(int route, bool skew) {
        auto& graph = *routes[route];
        select(0);
        graph.definition[skew].capture(execution.dev[0]->stream, [&] {
            for (int call = 0; call < kCalls; ++call) {
                if (skew && call % 16 == 0) {
                    const int delayed = (call / 16) % 2;
                    select(delayed);
                    producer_skew<<<1, 1, 0, execution.dev[delayed]->stream>>>(kSkewCycles);
                    CUDA_CHECK(cudaGetLastError());
                }
                for (int rank = 0; rank < 2; ++rank) {
                    select(rank);
                    auto* local = reinterpret_cast<ops::detail::PeerVecBf16*>(
                        static_cast<std::uint8_t*>(ranks[rank]->work[route]->data) +
                        call * stride + kGuard);
                    auto* arrival = reinterpret_cast<std::uint32_t*>(
                        static_cast<std::uint8_t*>(ranks[rank]->arrivals[route]->data) + kGuard);
                    ops::detail::peer_exchange_sum_kernel
                        <<<blocks(route), 256, 0, execution.dev[rank]->stream>>>(
                            local, local,
                            reinterpret_cast<ops::detail::PeerVecBf16*>(graph.mailbox.payload(rank, call)),
                            graph.mailbox.flag(rank, call),
                            reinterpret_cast<const ops::detail::PeerVecBf16*>(graph.mailbox.payload(1-rank, call)),
                            graph.mailbox.flag(1-rank, call), arrival + call,
                            graph.mailbox.hang(), bytes / 16);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            select(0);
        }, {&graph.bridge, execution.dev[1]->stream});
        graph.executable[skew].instantiate(graph.definition[skew]);
    }
    void set_epoch(int epoch) {
        retire();
        for (auto& image : ingress) { std::fill(image.begin(), image.end(), kCanary); }
        std::fill(expected.begin(), expected.end(), kCanary);
        for (int call = 0; call < kCalls; ++call) {
            for (int element = 0; element < elements; ++element) {
                const auto pair = operands(epoch, call, element);
                const auto offset = call * stride + kGuard + element * sizeof(std::uint16_t);
                for (int rank = 0; rank < 2; ++rank) {
                    std::memcpy(ingress[rank].data() + offset, &pair[rank], sizeof(pair[rank]));
                }
                const std::uint16_t sum = encode_bf16(decode_bf16(pair[0]) + decode_bf16(pair[1]));
                std::memcpy(expected.data() + offset, &sum, sizeof(sum));
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaMemcpy(ranks[rank]->seed.data, ingress[rank].data(), image_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    void prepare(int route, bool poison) {
        retire();
        routes[route]->mailbox.reset(poison);
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaMemcpyAsync(ranks[rank]->work[route]->data, ranks[rank]->seed.data,
                                       image_bytes, cudaMemcpyDeviceToDevice, execution.dev[rank]->stream));
        }
        // This also proves the peer input restore completes BEFORE replay begins. The timed
        // graph then contains only exchanges and the explicitly reported optional skew.
        retire();
    }
    Sample replay(int route, bool skew) {
        select(0);
        const auto start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(begin, execution.dev[0]->stream));
        routes[route]->executable[skew].launch(execution.dev[0]->stream);
        const auto enqueued = std::chrono::steady_clock::now();
        // DecodeGraphPeerBridge's captured join makes this end event depend on BOTH ranks.
        CUDA_CHECK(cudaEventRecord(end, execution.dev[0]->stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        const auto completed = std::chrono::steady_clock::now();
        // Every replay must validate its retired fault before the next prepare() can clear it.
        // A failed warmup/intermediate timed round must never become a fast accepted sample.
        if (*routes[route]->mailbox.hang() != 0) {
            throw std::runtime_error(std::string(kRouteNames[route]) + " mailbox timeout after replay");
        }
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
        const auto us = [](auto delta) {
            return std::chrono::duration<double, std::micro>(delta).count();
        };
        return {us(enqueued-start), us(completed-start), milliseconds * 1000.0};
    }
    void verify(int route, std::string_view phase) {
        retire();
        const std::string label = std::string(phase) + " " + kRouteNames[route];
        const auto& mailbox = routes[route]->mailbox;
        if (*mailbox.hang() != 0) { throw std::runtime_error(label + " mailbox timeout"); }
        for (int rank = 0; rank < 2; ++rank) {
            const std::string rank_label = label + " rank" + std::to_string(rank);
            const auto seed = ranks[rank]->seed.read();
            const auto work = ranks[rank]->work[route]->read();
            const auto arrival = ranks[rank]->arrivals[route]->read();
            require_bytes(seed.data(), ingress[rank].data(), image_bytes, rank_label + " immutable input");
            // Covers EVERY call's entire output and all device prefix/suffix/inter-slot guards.
            require_bytes(work.data(), expected.data(), image_bytes, rank_label + " FP64 oracle/raw BF16");
            require_canary(arrival.data(), kGuard, rank_label + " arrival prefix");
            require_canary(arrival.data()+kGuard+kCalls*4, kGuard, rank_label + " arrival suffix");
            for (int call = 0; call < kCalls; ++call) {
                std::uint32_t counter = 0;
                std::memcpy(&counter, arrival.data()+kGuard+call*4, sizeof(counter));
                if (counter != 0 || *mailbox.flag(rank, call) != 1) {
                    throw std::runtime_error(rank_label + " publication/arrival incomplete at call " + std::to_string(call));
                }
                const auto* payload = mailbox.payload(rank, call);
                require_bytes(payload, ingress[rank].data()+call*stride+kGuard, bytes,
                              rank_label + " published input call" + std::to_string(call));
                require_canary(payload-kGuard, kGuard, rank_label + " host payload prefix");
                require_canary(payload+bytes, kGuard, rank_label + " host payload suffix");
            }
            const auto* flag_region = mailbox.base+2*mailbox.payload_span+rank*mailbox.flag_span;
            require_canary(flag_region, kGuard, rank_label + " flag prefix");
            require_canary(flag_region+kGuard+kCalls*4, mailbox.flag_span-kGuard-kCalls*4,
                           rank_label + " flag suffix");
        }
        require_canary(mailbox.base+mailbox.hang_offset, kGuard, label + " fault prefix");
        require_canary(mailbox.base+mailbox.hang_offset+kGuard+4, 2*kGuard-4, label + " fault suffix");
    }
    void compare_routes() const {
        for (int rank = 0; rank < 2; ++rank) {
            const auto first = ranks[rank]->work[0]->read();
            const auto second = ranks[rank]->work[1]->read();
            require_bytes(second.data(), first.data(), image_bytes, "exact candidate/baseline bits");
        }
    }
};

double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(q * (values.size()-1))];
}
void run_case(const Options& options, int bytes) {
    Fixture fixture(options.devices, bytes);
    // Changing per-call inputs and three changed epochs exercise re-publication and stale-flag
    // hazards. Every configuration is replayed, with both early/late producer ranks, before timing.
    for (int epoch = 0; epoch < 3; ++epoch) {
        fixture.set_epoch(epoch);
        for (int skew = 0; skew < 2; ++skew) {
            for (int order = 0; order < 2; ++order) {
                const int route = (order + epoch + skew) % 2;
                fixture.prepare(route, true);
                (void)fixture.replay(route, skew != 0);
                fixture.verify(route, "qualification");
            }
            fixture.compare_routes();
        }
    }
    std::cout << "{\"type\":\"qualification\",\"bytes_per_rank\":" << bytes
              << ",\"calls\":" << kCalls << ",\"epochs\":3,\"all_elements\":true,"
                 "\"raw_bf16_exact\":true,\"mapped_publication_verified\":true,\"passed\":true}\n";
    if (options.qualify_only) { return; }
    for (int skew = 0; skew < 2; ++skew) {
        for (int warmup = 0; warmup < options.warmup; ++warmup) {
            for (int order = 0; order < 2; ++order) {
                const int route = (order+warmup) % 2;
                fixture.prepare(route, false);
                (void)fixture.replay(route, skew != 0);
            }
        }
        std::array<std::vector<double>, 2> joined, wall, enqueue;
        std::vector<double> paired_ratios;
        for (int pair = 0; pair < options.samples; ++pair) {
            std::array<Sample, 2> result;
            for (int order = 0; order < 2; ++order) {
                const int route = (order+pair+skew) % 2;
                fixture.prepare(route, false);
                result[route] = fixture.replay(route, skew != 0);
                joined[route].push_back(result[route].joined_us);
                wall[route].push_back(result[route].wall_us);
                enqueue[route].push_back(result[route].enqueue_us);
                std::cout << "{\"type\":\"sample\",\"bytes_per_rank\":" << bytes
                          << ",\"skew\":" << skew << ",\"pair\":" << pair
                          << ",\"route\":\"" << kRouteNames[route] << "\",\"blocks\":" << fixture.blocks(route)
                          << ",\"host_enqueue_us\":" << result[route].enqueue_us
                          << ",\"complete_wall_us\":" << result[route].wall_us
                          << ",\"joined_device_us\":" << result[route].joined_us << "}\n";
            }
            paired_ratios.push_back(result[1].joined_us / result[0].joined_us);
        }
        // Inspect each route's resident LAST TIMED output before any poison/reset/reissue.
        for (int route = 0; route < 2; ++route) { fixture.verify(route, "timed-final"); }
        fixture.compare_routes();
        for (int route = 0; route < 2; ++route) {
            std::cout << "{\"type\":\"summary\",\"bytes_per_rank\":" << bytes
                      << ",\"skew\":" << skew << ",\"calls\":" << kCalls
                      << ",\"route\":\"" << kRouteNames[route] << "\",\"blocks\":" << fixture.blocks(route)
                      << ",\"median_joined_device_us\":" << quantile(joined[route], 0.5)
                      << ",\"p95_joined_device_us\":" << quantile(joined[route], 0.95)
                      << ",\"median_complete_wall_us\":" << quantile(wall[route], 0.5)
                      << ",\"median_host_enqueue_us\":" << quantile(enqueue[route], 0.5)
                      << ",\"median_joined_us_per_exchange\":" << quantile(joined[route], 0.5)/kCalls
                      << ",\"qualified_after_timing\":true}\n";
        }
        std::cout << "{\"type\":\"comparison\",\"bytes_per_rank\":" << bytes
                  << ",\"skew\":" << skew << ",\"pairs\":" << options.samples
                  << ",\"median_paired_joined_ratio_one_over_production\":" << quantile(paired_ratios, 0.5)
                  << "}\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count < 2)) {
            std::cout << "SKIP: two CUDA devices required\n"; return 77;
        }
        CUDA_CHECK(status);
        if (options.devices[0] >= count || options.devices[1] >= count) {
            throw std::invalid_argument("selected device does not exist");
        }
        std::cout << std::setprecision(10)
                  << "{\"type\":\"configuration\",\"devices\":[" << options.devices[0] << ',' << options.devices[1]
                  << "],\"samples\":" << options.samples << ",\"warmup_replays\":" << options.warmup
                  << ",\"calls_per_graph\":" << kCalls << ",\"threads\":256,\"group\":4,"
                     "\"scope\":\"homogeneous captured mailbox exchange chain; not engine latency\","
                     "\"timed_input_restore\":false,\"timed_flag_reset\":false,"
                     "\"skew_cycles\":" << kSkewCycles << ",\"skew_every_calls\":16,"
                     "\"skew_rank\":\"alternating\",\"order\":\"alternating paired\","
                     "\"production_dispatch_changed\":false}\n";
        run_case(options, 10240);
        run_case(options, 40960);
        std::cout << "{\"type\":\"result\",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL mailbox geometry: " << error.what() << '\n';
        return 1;
    }
}
