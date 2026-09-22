// Private eager transport experiment: identical full-buffer BF16 sum, tiled host DMA only.
// No production route is changed. Qualification retains every changing call before timing zeros.
#include "ops/peer_transfer_fixture.h"
#include "ops/launcher/residual_add.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr std::size_t bytes = 10u << 20;
constexpr int elements = static_cast<int>(bytes / 2);
constexpr int qualification_calls = 4;
constexpr std::size_t guard_bytes = 256;
constexpr unsigned char canary = PeerTransferFixture::host_canary;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Explicit Program-like owner. Bound main streams outlive this owner. Allocation, capture,
// fallback dispatch, and CPU synchronization are absent from the timed candidate itself.
class Pipeline {
public:
    Pipeline(PeerTransferFixture& fixture, int tiles) : fixture_(fixture), tiles_(tiles) {
        require(tiles == 2 || tiles == 4, "candidate admits exactly two or four tiles");
        try {
            for (int rank = 0; rank < 2; ++rank) {
                fixture_.select(rank);
                CUDA_CHECK(cudaStreamCreateWithFlags(&copy_[rank], cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&producer_[rank], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&pulled_[rank], cudaEventDisableTiming));
                for (int tile = 0; tile < tiles_; ++tile) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&ready_[rank][tile], cudaEventDisableTiming));
                }
                CUDA_CHECK(cudaHostAlloc(&allocation_[rank], bytes + 2 * guard_bytes,
                                         cudaHostAllocPortable));
            }
            reset_host();
        } catch (...) { cleanup(); throw; }
    }
    ~Pipeline() { cleanup(); }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void* host(int rank) const {
        return static_cast<unsigned char*>(allocation_[rank]) + guard_bytes;
    }
    void reset_host() {
        for (void* allocation : allocation_) {
            std::memset(allocation, canary, bytes + 2 * guard_bytes);
        }
    }
    int host_guards() const {
        for (int rank = 0; rank < 2; ++rank) {
            const auto* p = static_cast<const unsigned char*>(allocation_[rank]);
            for (std::size_t i = 0; i < guard_bytes; ++i) {
                if (p[i] != canary || p[guard_bytes + bytes + i] != canary) {
                    std::cerr << "pipeline pinned prefix/suffix guard changed\n";
                    return 1;
                }
            }
        }
        return 0;
    }

    void sum() {
        // Same size/capture admission query as the public explicit-pinned route; reject before
        // touching auxiliary streams. This experiment has no captured or direct-P2P candidate.
        require(fixture_.transfer.uses_host_staging({bytes, bytes}),
                "pipeline is eager pinned only");
        constexpr std::size_t total_bytes = bytes;
        const std::size_t tile_bytes = total_bytes / static_cast<std::size_t>(tiles_);
        // A: publish every current event record on BOTH ranks before any cross-rank wait.
        // A wait binds the most recently issued record, not an intended future record.
        for (int rank = 0; rank < 2; ++rank) {
            fixture_.select(rank);
            const auto main = fixture_.execution.dev[rank]->stream;
            CUDA_CHECK(cudaEventRecord(producer_[rank], main));
            CUDA_CHECK(cudaStreamWaitEvent(copy_[rank], producer_[rank], 0));
            for (int tile = 0; tile < tiles_; ++tile) {
                const std::size_t offset = tile_bytes * static_cast<std::size_t>(tile);
                CUDA_CHECK(cudaMemcpyAsync(static_cast<unsigned char*>(host(rank)) + offset,
                    static_cast<const unsigned char*>(fixture_.storage[rank]->input.data()) + offset,
                    tile_bytes, cudaMemcpyDeviceToHost, copy_[rank]));
                CUDA_CHECK(cudaEventRecord(ready_[rank][tile], copy_[rank]));
            }
        }
        // B: peer tile N may arrive while local outgoing tile N+1 is still being copied.
        // Both complete inbound-copy records must exist before phase C's peer waits.
        for (int rank = 0; rank < 2; ++rank) {
            fixture_.select(rank);
            const auto main = fixture_.execution.dev[rank]->stream;
            for (int tile = 0; tile < tiles_; ++tile) {
                const std::size_t offset = tile_bytes * static_cast<std::size_t>(tile);
                CUDA_CHECK(cudaStreamWaitEvent(main, ready_[1 - rank][tile], 0));
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<unsigned char*>(fixture_.storage[rank]->staging.data()) + offset,
                    static_cast<const unsigned char*>(host(1 - rank)) + offset,
                    tile_bytes, cudaMemcpyHostToDevice, main));
            }
            CUDA_CHECK(cudaEventRecord(pulled_[rank], main));
        }
        // C: own final D2H protects the in-place source; peer H2D retirement protects this
        // rank's pinned buffer before the next producer gate. The sum kernel/geometry is
        // exactly the public full-buffer residual_add, not a tiled numerical reduction.
        for (int rank = 0; rank < 2; ++rank) {
            fixture_.select(rank);
            const auto main = fixture_.execution.dev[rank]->stream;
            CUDA_CHECK(cudaStreamWaitEvent(main, ready_[rank][tiles_ - 1], 0));
            CUDA_CHECK(cudaStreamWaitEvent(main, pulled_[1 - rank], 0));
            Tensor input(fixture_.storage[rank]->input.data(), DType::BF16, {elements});
            const Tensor staging(fixture_.storage[rank]->staging.data(), DType::BF16, {elements});
            ops::detail::residual_add_launch(staging, input, main);
        }
    }

private:
    void cleanup() noexcept {
        // Retire ALL four streams even after a partially enqueued call: a failed phase A can
        // leave auxiliary D2H work with no final main-stream join. No owner storage is global.
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(fixture_.execution.dev[rank]->device);
            (void)cudaStreamSynchronize(fixture_.execution.dev[rank]->stream);
            if (copy_[rank]) { (void)cudaStreamSynchronize(copy_[rank]); }
        }
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(fixture_.execution.dev[rank]->device);
            if (allocation_[rank]) { (void)cudaFreeHost(allocation_[rank]); }
            if (producer_[rank]) { (void)cudaEventDestroy(producer_[rank]); }
            if (pulled_[rank]) { (void)cudaEventDestroy(pulled_[rank]); }
            for (auto event : ready_[rank]) { if (event) { (void)cudaEventDestroy(event); } }
            if (copy_[rank]) { (void)cudaStreamDestroy(copy_[rank]); }
            allocation_[rank] = nullptr;
            producer_[rank] = pulled_[rank] = nullptr;
            ready_[rank].fill(nullptr);
            copy_[rank] = nullptr;
        }
    }
    PeerTransferFixture& fixture_;
    int tiles_;
    std::array<cudaStream_t, 2> copy_{};
    std::array<cudaEvent_t, 2> producer_{}, pulled_{};
    std::array<std::array<cudaEvent_t, 4>, 2> ready_{};
    std::array<void*, 2> allocation_{};
};

struct Route {
    const char* name;
    Pipeline* pipeline;
    void sum(PeerTransferFixture& fixture) const {
        if (pipeline) { pipeline->sum(); } else { fixture.sum(fixture.transfer); }
    }
    void* host(PeerTransferFixture& fixture, int rank) const {
        return pipeline ? pipeline->host(rank) : fixture.transfer.host_buffer(rank);
    }
    void reset_host(PeerTransferFixture& fixture) const {
        if (pipeline) { pipeline->reset_host(); }
        else {
            for (int rank = 0; rank < 2; ++rank) {
                std::memset(host(fixture, rank), canary, bytes + guard_bytes);
            }
        }
    }
    int host_guards(PeerTransferFixture& fixture) const {
        if (pipeline) { return pipeline->host_guards(); }
        for (int rank = 0; rank < 2; ++rank) {
            const auto* p = static_cast<const unsigned char*>(host(fixture, rank));
            for (std::size_t i = bytes; i < bytes + guard_bytes; ++i) {
                if (p[i] != canary) { std::cerr << "public pinned suffix changed\n"; return 1; }
            }
        }
        return 0;
    }
};

void prepare_nonzero(PeerTransferFixture& fixture, int epoch) {
    for (int rank = 0; rank < 2; ++rank) {
        auto* p = static_cast<std::uint16_t*>(fixture.storage[rank]->ingress.data());
        for (int call = 0; call < qualification_calls; ++call) {
            for (int i = 0; i < elements; ++i) {
                // Avalanche the FULL absolute index: a 4KiB periodic pattern would repeat at
                // every tile boundary and incorrectly accept copying tile zero repeatedly.
                std::uint32_t x = static_cast<std::uint32_t>(i) ^
                    (0x9e3779b9u * static_cast<std::uint32_t>(rank + 1)) ^
                    (0x85ebca6bu * static_cast<std::uint32_t>(call + 1)) ^
                    (0xc2b2ae35u * static_cast<std::uint32_t>(epoch + 1));
                x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
                const int value = static_cast<int>(x & 2047u) - 1024;
                p[static_cast<std::size_t>(call) * elements + i] =
                    f32_to_bf16(static_cast<float>(value) / 128.0f);
            }
        }
    }
}

int device_guards(PeerTransferFixture& fixture) {
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        for (const auto* buffer : {&fixture.storage[rank]->input, &fixture.storage[rank]->staging,
                                  &fixture.storage[rank]->sum_history,
                                  &fixture.storage[rank]->host_source_history,
                                  &fixture.storage[rank]->skew}) {
            failures += buffer->verify_guards("pipeline fixture");
        }
    }
    return failures;
}

int qualify(PeerTransferFixture& fixture, const Route& route, int epoch,
            const Route* alternate = nullptr) {
    fixture.retire();
    prepare_nonzero(fixture, epoch);
    route.reset_host(fixture);
    if (alternate) { alternate->reset_host(fixture); }
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        fixture.storage[rank]->sum_history.fill(0xcd);
        fixture.storage[rank]->host_source_history.fill(0xcd);
        fixture.storage[rank]->staging.fill(0xcd);
        CUDA_CHECK(cudaDeviceSynchronize()); // retire only initialization, before sequence
    }
    // No host synchronization between any of these changing calls. Both producer skew
    // directions are exercised; retained host-source snapshots also test reuse ordering.
    for (int call = 0; call < qualification_calls; ++call) {
        fixture.skew((call + epoch) % 2);
        fixture.upload(call);
        const Route& selected = alternate && call % 2 ? *alternate : route;
        selected.sum(fixture);
        for (int rank = 0; rank < 2; ++rank) {
            fixture.select(rank);
            const auto stream = fixture.execution.dev[rank]->stream;
            auto* result = static_cast<unsigned char*>(fixture.storage[rank]->sum_history.data())
                + static_cast<std::size_t>(call) * bytes;
            auto* publication = static_cast<unsigned char*>(fixture.storage[rank]->host_source_history.data())
                + static_cast<std::size_t>(call) * bytes;
            CUDA_CHECK(cudaMemcpyAsync(result, fixture.storage[rank]->input.data(), bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(publication, selected.host(fixture, rank), bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
    }
    fixture.retire();
    int failures = device_guards(fixture) + route.host_guards(fixture);
    if (alternate) { failures += alternate->host_guards(fixture); }
    for (int call = 0; call < qualification_calls; ++call) {
        // Existing independent FP64 oracle decodes the represented BF16 operands. The bounded
        // dyadic sum is exactly FP32-representable, so final BF16 RNE has no double-rounding gap.
        const auto expected = fixture.sum_oracle(call);
        for (int rank = 0; rank < 2; ++rank) {
            fixture.select(rank);
            const auto* result = static_cast<const unsigned char*>(fixture.storage[rank]->sum_history.data())
                + static_cast<std::size_t>(call) * bytes;
            const auto* publication = static_cast<const unsigned char*>(fixture.storage[rank]->host_source_history.data())
                + static_cast<std::size_t>(call) * bytes;
            const auto* source = static_cast<const std::uint16_t*>(fixture.storage[rank]->ingress.data())
                + static_cast<std::size_t>(call) * elements;
            failures += verify_exact("pipeline every output", from_device<std::uint16_t>(result, elements), expected);
            failures += verify_exact("pipeline every pinned publication",
                from_device<std::uint16_t>(publication, elements),
                std::vector<std::uint16_t>(source, source + elements));
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        const auto* peer = static_cast<const std::uint16_t*>(fixture.storage[1 - rank]->ingress.data())
            + static_cast<std::size_t>(qualification_calls - 1) * elements;
        failures += verify_exact("pipeline final incoming storage",
            from_device<std::uint16_t>(fixture.storage[rank]->staging.data(), elements),
            std::vector<std::uint16_t>(peer, peer + elements));
    }
    std::cout << "{\"type\":\"qualification\",\"mode\":\"" << route.name
              << (alternate ? "+public-alternating" : "")
              << "\",\"epoch\":" << epoch << ",\"devices\":[" << fixture.execution.dev[0]->device
              << ',' << fixture.execution.dev[1]->device << "],\"calls\":" << qualification_calls
              << ",\"failures\":" << failures << "}\n";
    return failures;
}

int pending_owner_case(PeerTransferFixture& fixture, int tiles) {
    fixture.retire();
    prepare_nonzero(fixture, 11 + tiles);
    {
        Pipeline owner(fixture, tiles);
        fixture.skew(tiles == 2 ? 0 : 1);
        fixture.upload(0);
        owner.sum();
        // Deliberately no host wait here. Owner teardown must retire its auxiliary streams
        // AND the borrowed main streams before freeing pinned storage/events.
    }
    const auto expected = fixture.sum_oracle(0);
    int failures = device_guards(fixture);
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        failures += verify_exact("pending-owner output survives teardown",
            from_device<std::uint16_t>(fixture.storage[rank]->input.data(), elements), expected);
    }
    std::cout << "{\"type\":\"owner-retirement\",\"tiles\":" << tiles
              << ",\"devices\":[" << fixture.execution.dev[0]->device << ','
              << fixture.execution.dev[1]->device << "],\"failures\":" << failures << "}\n";
    return failures;
}

class Timers {
public:
    explicit Timers(PeerTransferFixture& fixture) : fixture_(fixture) {
        try {
            for (int rank = 0; rank < 2; ++rank) {
                fixture_.select(rank);
                CUDA_CHECK(cudaEventCreate(&begin[rank]));
                CUDA_CHECK(cudaEventCreate(&end[rank]));
            }
            fixture_.select(0);
            CUDA_CHECK(cudaEventCreate(&joined));
        } catch (...) { cleanup(); throw; }
    }
    ~Timers() { cleanup(); }
    Timers(const Timers&) = delete;
    Timers& operator=(const Timers&) = delete;
    std::array<cudaEvent_t, 2> begin{}, end{};
    cudaEvent_t joined = nullptr;
private:
    void cleanup() noexcept {
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(fixture_.execution.dev[rank]->device);
            if (begin[rank]) { (void)cudaEventDestroy(begin[rank]); }
            if (end[rank]) { (void)cudaEventDestroy(end[rank]); }
        }
        (void)cudaSetDevice(fixture_.execution.dev[0]->device);
        if (joined) { (void)cudaEventDestroy(joined); }
    }
    PeerTransferFixture& fixture_;
};

int verify_timed(PeerTransferFixture& fixture, const Route& route) {
    const std::vector<std::uint16_t> zero(elements, 0);
    int failures = device_guards(fixture) + route.host_guards(fixture);
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        failures += verify_exact("resident timed output",
            from_device<std::uint16_t>(fixture.storage[rank]->input.data(), elements), zero);
        failures += verify_exact("resident timed incoming storage",
            from_device<std::uint16_t>(fixture.storage[rank]->staging.data(), elements), zero);
        const auto* p = static_cast<const std::uint16_t*>(route.host(fixture, rank));
        failures += verify_exact("resident timed pinned publication",
            std::vector<std::uint16_t>(p, p + elements), zero);
    }
    return failures;
}

int measure(PeerTransferFixture& fixture, const Route& route, Timers& timer,
            int calls, int sample, int position) {
    fixture.upload(0); // immutable zeros, outside measured interval
    fixture.retire();
    fixture.select(0);
    CUDA_CHECK(cudaEventRecord(timer.begin[0], fixture.execution.dev[0]->stream));
    fixture.select(1);
    CUDA_CHECK(cudaStreamWaitEvent(fixture.execution.dev[1]->stream, timer.begin[0], 0));
    CUDA_CHECK(cudaEventRecord(timer.begin[1], fixture.execution.dev[1]->stream));
    const auto begin = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; ++call) { route.sum(fixture); }
    const auto enqueued = std::chrono::steady_clock::now();
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        CUDA_CHECK(cudaEventRecord(timer.end[rank], fixture.execution.dev[rank]->stream));
    }
    fixture.select(0);
    CUDA_CHECK(cudaStreamWaitEvent(fixture.execution.dev[0]->stream, timer.end[1], 0));
    CUDA_CHECK(cudaEventRecord(timer.joined, fixture.execution.dev[0]->stream));
    CUDA_CHECK(cudaEventSynchronize(timer.joined));
    const auto completed = std::chrono::steady_clock::now();
    float joined_ms = 0;
    std::array<float, 2> rank_ms{};
    CUDA_CHECK(cudaEventElapsedTime(&joined_ms, timer.begin[0], timer.joined));
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        CUDA_CHECK(cudaEventElapsedTime(&rank_ms[rank], timer.begin[rank], timer.end[rank]));
    }
    // Check each route's resident results before the next reset can hide corruption.
    const int failures = verify_timed(fixture, route);
    std::cout << std::setprecision(9) << "{\"type\":\"sample\",\"sample\":" << sample
              << ",\"position\":" << position << ",\"mode\":\"" << route.name
              << "\",\"devices\":[" << fixture.execution.dev[0]->device << ','
              << fixture.execution.dev[1]->device << "],\"bytes_per_rank\":" << bytes
              << ",\"collectives\":" << calls
              << ",\"host_enqueue_ms\":" << std::chrono::duration<double, std::milli>(enqueued - begin).count()
              << ",\"complete_wall_ms\":" << std::chrono::duration<double, std::milli>(completed - begin).count()
              << ",\"joined_device_ms\":" << joined_ms
              << ",\"rank_device_ms\":[" << rank_ms[0] << ',' << rank_ms[1]
              << "],\"failures\":" << failures << "}\n";
    return failures;
}

int run_order(const std::vector<int>& devices, int calls, int samples) {
    PeerTransferFixture fixture(devices, elements, qualification_calls,
                                {elements, elements}, bytes + guard_bytes);
    Pipeline two(fixture, 2), four(fixture, 4);
    const std::array<Route, 3> routes{{{"public-pinned", nullptr}, {"pipeline-2", &two}, {"pipeline-4", &four}}};
    Timers timers(fixture);
    require(fixture.transfer.uses_host_staging({bytes, bytes}), "public pinned route inactive");
    int failures = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
        for (const auto& route : routes) { failures += qualify(fixture, route, epoch); }
    }
    // Cross-route reuse matters when a full prefill reduction is followed by a retained
    // public route: neither transport may leave the shared caller buffers in flight.
    failures += qualify(fixture, routes[1], 5, &routes[0]);
    failures += qualify(fixture, routes[2], 6, &routes[0]);
    failures += pending_owner_case(fixture, 2);
    failures += pending_owner_case(fixture, 4);
    require(failures == 0, "changing-input qualification failed; timing rejected");
    // Each repeated in-place sum starts and remains zero, avoiding overflow and preserving
    // the qualified instruction/transfer workload. Uploads are never included in timing.
    for (int rank = 0; rank < 2; ++rank) {
        std::memset(fixture.storage[rank]->ingress.data(), 0, bytes);
    }
    for (const auto& route : routes) {
        fixture.upload(0);
        fixture.retire();
        for (int call = 0; call < 8; ++call) { route.sum(fixture); }
        fixture.retire();
        failures += verify_timed(fixture, route);
    }
    require(failures == 0, "warmup failed; timing rejected");
    for (int sample = 0; sample < samples; ++sample) {
        // Rotate each route through every position; reverse direction every full rotation.
        for (int position = 0; position < 3; ++position) {
            const int signed_position = (sample / 3) % 2 == 0 ? position : 2 - position;
            const int index = (sample % 3 + signed_position) % 3;
            failures += measure(fixture, routes[index], timers, calls, sample, position);
            require(failures == 0, "timed resident output/guard failure");
        }
    }
    // Re-qualify with fresh nonzero inputs after timing so zero-only no-ops cannot qualify.
    for (const auto& route : routes) { failures += qualify(fixture, route, 7); }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        int calls = 32, samples = 31;
        if (argc != 1 && argc != 3) {
            std::cerr << "usage: ninfer_peer_transfer_pipeline_bench [collectives-per-batch paired-samples]\n";
            return 2;
        }
        if (argc == 3) {
            calls = std::stoi(argv[1]); samples = std::stoi(argv[2]);
            require(calls > 0 && samples > 0, "positive timing counts required");
        }
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count < 2) { std::cout << "SKIP: two CUDA devices required\n"; return 77; }
        int forward = 0, reverse = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, 0, 1));
        CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, 1, 0));
        if (forward || reverse) { std::cout << "SKIP: candidate requires no P2P in either direction\n"; return 77; }
        std::cout << "{\"type\":\"configuration\",\"scope\":\"eager 10MiB allreduce; no engine speed claim\","
                     "\"routes\":[\"public-pinned\",\"pipeline-2\",\"pipeline-4\"],"
                     "\"order\":\"rotating paired, reversed each three samples\","
                     "\"warmup_collectives_per_route\":8,\"qualification_calls_per_epoch\":4,"
                     "\"producer_skew\":\"alternating rank during qualification only\","
                     "\"arithmetic\":\"unchanged full-buffer residual_add\",\"devices\":[";
        for (int device = 0; device < 2; ++device) {
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            if (device) { std::cout << ','; }
            std::cout << "{\"id\":" << device << ",\"name\":\"" << prop.name
                      << "\",\"sm_count\":" << prop.multiProcessorCount
                      << ",\"compute_capability\":[" << prop.major << ',' << prop.minor
                      << "],\"asyncEngineCount\":" << prop.asyncEngineCount
                      << ",\"concurrentKernels\":" << prop.concurrentKernels
                      << ",\"unifiedAddressing\":" << prop.unifiedAddressing
                      << ",\"tccDriver\":" << prop.tccDriver << '}';
        }
        std::cout << "]}\n";
        int failures = run_order({0, 1}, calls, samples);
        failures += run_order({1, 0}, calls, samples);
        std::cout << "{\"type\":\"result\",\"failures\":" << failures << "}\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL peer transfer pipeline: " << error.what() << '\n';
        return 1;
    }
}
