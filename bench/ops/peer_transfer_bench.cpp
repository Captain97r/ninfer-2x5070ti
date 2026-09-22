// Paired eager collective timing: fixed buffers, identical BF16 sum, no allocation inside a batch.
// Reports host enqueue separately from complete wall time and a cross-device event join.
#include "ops/peer_transfer_fixture.h"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;

namespace {

class Timers {
public:
    const ExecutionContext& execution;
    std::array<cudaEvent_t, 2> begin{nullptr, nullptr};
    std::array<cudaEvent_t, 2> end{nullptr, nullptr};
    cudaEvent_t joined = nullptr;
    explicit Timers(const ExecutionContext& ec) : execution(ec) {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            CUDA_CHECK(cudaEventCreate(&begin[rank]));
            CUDA_CHECK(cudaEventCreate(&end[rank]));
        }
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        CUDA_CHECK(cudaEventCreate(&joined));
    }
    ~Timers() {
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            (void)cudaEventDestroy(begin[rank]);
            (void)cudaEventDestroy(end[rank]);
        }
        (void)cudaSetDevice(execution.dev[0]->device);
        (void)cudaEventDestroy(joined);
    }
};

void measure(PeerTransferFixture& fixture, const ops::PeerTransfer& selected,
             Timers& timer, int calls, int pair, const char* mode) {
    // Zeros remain finite through arbitrarily many sums. Mathematical qualification uses
    // changing nonzero operands in the separate exact test, not this timing workload.
    fixture.upload(0);
    fixture.retire();
    fixture.select(0);
    CUDA_CHECK(cudaEventRecord(timer.begin[0], fixture.execution.dev[0]->stream));
    fixture.select(1);
    CUDA_CHECK(cudaStreamWaitEvent(fixture.execution.dev[1]->stream, timer.begin[0], 0));
    CUDA_CHECK(cudaEventRecord(timer.begin[1], fixture.execution.dev[1]->stream));

    const auto begin = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; ++call) { fixture.sum(selected); }
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

    float joined_ms = 0.0f;
    std::array<float, 2> rank_ms{};
    CUDA_CHECK(cudaEventElapsedTime(&joined_ms, timer.begin[0], timer.joined));
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        CUDA_CHECK(cudaEventElapsedTime(&rank_ms[rank], timer.begin[rank], timer.end[rank]));
    }
    const double enqueue_ms = std::chrono::duration<double, std::milli>(enqueued - begin).count();
    const double complete_ms = std::chrono::duration<double, std::milli>(completed - begin).count();
    std::cout << std::setprecision(9)
              << "{\"type\":\"sample\",\"pair\":" << pair
              << ",\"mode\":\"" << mode << "\",\"bytes_per_rank\":" << fixture.elements * 2
              << ",\"collectives\":" << calls
              << ",\"host_enqueue_ms\":" << enqueue_ms
              << ",\"complete_wall_ms\":" << complete_ms
              << ",\"joined_device_ms\":" << joined_ms
              << ",\"rank_device_ms\":[" << rank_ms[0] << ',' << rank_ms[1] << "]}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        int calls = 128;
        int pairs = 5;
        if (argc != 1 && argc != 3) {
            std::cerr << "usage: ninfer_peer_transfer_bench [collectives-per-batch pairs]\n";
            return 2;
        }
        if (argc == 3) {
            calls = std::stoi(argv[1]);
            pairs = std::stoi(argv[2]);
            if (calls <= 0 || pairs <= 0) { throw std::invalid_argument("counts must be positive"); }
        }
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count < 2) { std::cout << "SKIP: two devices required\n"; return 77; }
        int forward = 0;
        int reverse = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, 0, 1));
        CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, 1, 0));
        if (forward && reverse) {
            std::cout << "SKIP: implicit-versus-pinned benchmark requires no direct P2P\n";
            return 77;
        }
        std::cout << "{\"type\":\"configuration\",\"devices\":[0,1],"
                     "\"direct_p2p\":false,\"order\":\"alternating paired\","
                     "\"warmup_collectives_per_route\":8,\"timing\":\"eager unprofiled\"}\n";
        int failures = 0;
        for (const int bytes : {65536, 1048576, 10485760}) {
            PeerTransferFixture fixture({0, 1}, bytes / 2, 1,
                                          {bytes / 2, bytes / 2}, bytes + 256);
            ops::PeerTransfer implicit(fixture.execution);
            Timers timers(fixture.execution);
            if (!fixture.transfer.uses_host_staging(
                    {static_cast<std::size_t>(bytes), static_cast<std::size_t>(bytes)})) {
                throw std::runtime_error("paired benchmark explicit route is inactive");
            }
            for (int rank = 0; rank < 2; ++rank) {
                std::memset(fixture.storage[rank]->ingress.data(), 0,
                            fixture.storage[rank]->ingress.size());
            }
            fixture.upload(0);
            fixture.retire();
            for (int i = 0; i < 8; ++i) { fixture.sum(implicit); }
            fixture.retire();
            for (int i = 0; i < 8; ++i) { fixture.sum(fixture.transfer); }
            fixture.retire();
            for (int pair = 0; pair < pairs; ++pair) {
                if (pair % 2 == 0) {
                    measure(fixture, implicit, timers, calls, pair, "implicit");
                    measure(fixture, fixture.transfer, timers, calls, pair, "explicit-pinned");
                } else {
                    measure(fixture, fixture.transfer, timers, calls, pair, "explicit-pinned");
                    measure(fixture, implicit, timers, calls, pair, "implicit");
                }
            }
            const std::vector<std::uint16_t> zero(static_cast<std::size_t>(bytes) / 2, 0);
            for (int rank = 0; rank < 2; ++rank) {
                fixture.select(rank);
                failures += verify_exact("timed sum remains zero",
                    from_device<std::uint16_t>(fixture.storage[rank]->input.data(), zero.size()), zero);
                failures += fixture.storage[rank]->input.verify_guards("timed input");
                failures += fixture.storage[rank]->staging.verify_guards("timed staging");
            }
        }
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL peer transfer benchmark: " << error.what() << '\n';
        return 1;
    }
}
