// Public eager transport comparison: Automatic versus WholeBuffer scheduling of the same Op.
// Qualification retains every changing call before timing zeros; no private transport is measured.
#include "ops/peer_transfer_fixture.h"

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

struct Route {
    const char* name;
    const ops::PeerTransfer& transfer;
    bool expected_pipeline;

    void validate() const {
        require(transfer.uses_host_staging({bytes, bytes}), "public pinned route inactive");
        require(transfer.uses_two_tile_allreduce(bytes) == expected_pipeline,
                "public transport schedule does not select the expected route");
    }
    void sum(PeerTransferFixture& fixture) const { fixture.sum(transfer); }
    void* host(int rank) const { return transfer.host_buffer(rank); }
    void reset_host() const {
        for (int rank = 0; rank < 2; ++rank) {
            std::memset(host(rank), canary, transfer.host_capacity_bytes());
        }
    }
    int host_guards() const {
        for (int rank = 0; rank < 2; ++rank) {
            const auto* p = static_cast<const unsigned char*>(host(rank));
            for (std::size_t i = bytes; i < transfer.host_capacity_bytes(); ++i) {
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
    route.validate();
    route.reset_host();
    if (alternate) { alternate->validate(); alternate->reset_host(); }
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
            CUDA_CHECK(cudaMemcpyAsync(publication, selected.host(rank), bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
    }
    fixture.retire();
    int failures = device_guards(fixture) + route.host_guards();
    if (alternate) { failures += alternate->host_guards(); }
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
              << (alternate ? std::string("+") + alternate->name + "-alternating" : "")
              << "\",\"epoch\":" << epoch << ",\"devices\":[" << fixture.execution.dev[0]->device
              << ',' << fixture.execution.dev[1]->device << "],\"calls\":" << qualification_calls
              << ",\"failures\":" << failures << "}\n";
    return failures;
}

int pending_owner_case(PeerTransferFixture& fixture, ops::PeerTransferSchedule schedule,
                       const char* name) {
    const bool pipeline = schedule == ops::PeerTransferSchedule::Automatic;
    fixture.retire();
    prepare_nonzero(fixture, pipeline ? 13 : 15);
    {
        ops::PeerTransfer owner(fixture.execution, bytes + guard_bytes, schedule);
        require(owner.uses_two_tile_allreduce(bytes) == pipeline,
                "pending owner selected the wrong transport schedule");
        fixture.skew(pipeline ? 0 : 1);
        fixture.upload(0);
        fixture.sum(owner);
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
    std::cout << "{\"type\":\"owner-retirement\",\"mode\":\"" << name << "\""
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
    int failures = device_guards(fixture) + route.host_guards();
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        failures += verify_exact("resident timed output",
            from_device<std::uint16_t>(fixture.storage[rank]->input.data(), elements), zero);
        failures += verify_exact("resident timed incoming storage",
            from_device<std::uint16_t>(fixture.storage[rank]->staging.data(), elements), zero);
        const auto* p = static_cast<const std::uint16_t*>(route.host(rank));
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
                                {elements, elements}, 0);
    // Both measured owners outlive their queued work but retire BEFORE fixture device storage
    // is released, including partial-enqueue unwinding. Its unprovisioned owner is unused.
    // Omit the policy for Automatic, exactly as the Program does.
    ops::PeerTransfer automatic(fixture.execution, bytes + guard_bytes);
    if (!automatic.uses_two_tile_allreduce(bytes)) {
        std::cout << "SKIP: Automatic two-tile transport is not qualified on this configuration\n";
        return 77;
    }
    ops::PeerTransfer whole(fixture.execution, bytes + guard_bytes,
                            ops::PeerTransferSchedule::WholeBuffer);
    const std::array<Route, 2> routes{{{"whole-buffer", whole, false},
                                      {"automatic", automatic, true}}};
    Timers timers(fixture);
    for (const auto& route : routes) { route.validate(); }
    for (int rank = 0; rank < 2; ++rank) {
        require(whole.host_buffer(rank) != automatic.host_buffer(rank),
                "comparison resources unexpectedly share pinned storage");
    }
    std::cout << "{\"type\":\"route-selection\",\"devices\":[" << devices[0] << ',' << devices[1]
              << "],\"whole_buffer_two_tile\":" << (whole.uses_two_tile_allreduce(bytes) ? "true" : "false")
              << ",\"automatic_two_tile\":" << (automatic.uses_two_tile_allreduce(bytes) ? "true" : "false")
              << "}\n";
    int failures = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
        for (const auto& route : routes) { failures += qualify(fixture, route, epoch); }
    }
    // Neither schedule may leave shared caller buffers in flight when the other owner is
    // used next. Reverse the initial owner as well as alternating within each sequence.
    failures += qualify(fixture, routes[1], 5, &routes[0]);
    failures += qualify(fixture, routes[0], 6, &routes[1]);
    failures += pending_owner_case(fixture, ops::PeerTransferSchedule::Automatic, "automatic");
    failures += pending_owner_case(fixture, ops::PeerTransferSchedule::WholeBuffer, "whole-buffer");
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
        // Alternate order within each matched two-route sample.
        for (int position = 0; position < 2; ++position) {
            const int index = (sample + position) % 2;
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
                     "\"routes\":[\"whole-buffer\",\"automatic\"],"
                     "\"order\":\"alternating paired\","
                     "\"warmup_collectives_per_route\":8,\"qualification_calls_per_epoch\":4,"
                     "\"producer_skew\":\"alternating rank during qualification only\","
                     "\"arithmetic\":\"public allreduce_sum, identical full-buffer residual_add\",\"devices\":[";
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
        if (failures == 77) { return 77; }
        const int reversed = run_order({1, 0}, calls, samples);
        if (reversed == 77) { return 77; }
        failures += reversed;
        std::cout << "{\"type\":\"result\",\"failures\":" << failures << "}\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL peer transfer pipeline: " << error.what() << '\n';
        return 1;
    }
}
