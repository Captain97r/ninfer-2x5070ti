#include "ops/peer_transfer_fixture.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr std::size_t bytes = 10u << 20;
constexpr int elements = static_cast<int>(bytes / 2);
constexpr int calls = 5;
constexpr int partial_elements = 65539;
constexpr std::array<int, 2> gather_rows{32771, 49157};
constexpr int gathered_elements = gather_rows[0] + gather_rows[1];
constexpr std::size_t guard_bytes = 256;
using Schedule = ops::PeerTransferSchedule;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

const std::uint16_t* source(const PeerTransferFixture& fixture, int rank, int call) {
    return static_cast<const std::uint16_t*>(fixture.storage[rank]->ingress.data()) +
           static_cast<std::size_t>(call) * elements;
}

void prepare(PeerTransferFixture& fixture, int epoch) {
    fixture.retire();
    for (int rank = 0; rank < 2; ++rank) {
        auto* input = static_cast<std::uint16_t*>(fixture.storage[rank]->ingress.data());
        for (int call = 0; call < calls; ++call) {
            for (int i = 0; i < elements; ++i) {
                // Include the whole absolute index: periodic 4 KiB fixtures cannot expose a
                // wrong 5 MiB tile offset. All sum operands are bounded represented dyadics.
                std::uint32_t x = static_cast<std::uint32_t>(i) ^
                    (0x9e3779b9u * static_cast<std::uint32_t>(rank + 1)) ^
                    (0x85ebca6bu * static_cast<std::uint32_t>(call + 1)) ^
                    (0xc2b2ae35u * static_cast<std::uint32_t>(epoch + 1));
                x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
                input[static_cast<std::size_t>(call) * elements + i] =
                    f32_to_bf16(static_cast<float>(static_cast<int>(x & 2047u) - 1024) / 128.0f);
            }
        }
        // Call 3 is ONLY an exact gather: preserve NaN payloads, infinities and signed zero.
        constexpr std::array<std::uint16_t, 8> patterns{
            0x0000, 0x8000, 0x7fc1, 0xffc2, 0x7f80, 0xff80, 0x0001, 0x8001};
        for (int i = 0; i < gathered_elements; i += 4096) {
            input[3u * elements + i] = patterns[(i / 4096 + rank + epoch) % patterns.size()];
        }
        std::memset(fixture.transfer.host_buffer(rank), PeerTransferFixture::host_canary,
                    fixture.transfer.host_capacity_bytes());
        fixture.select(rank);
        fixture.storage[rank]->sum_history.fill(0xcd);
        fixture.storage[rank]->host_source_history.fill(0xcd);
        fixture.storage[rank]->gather_history.fill(0xcd);
        fixture.storage[rank]->staging.fill(0xcd);
        CUDA_CHECK(cudaDeviceSynchronize()); // default-stream initialization only
    }
}

void sum(PeerTransferFixture& fixture, const ops::PeerTransfer& transfer, int count) {
    const std::array<Tensor, 2> input{
        Tensor(fixture.storage[0]->input.data(), DType::BF16, {count}),
        Tensor(fixture.storage[1]->input.data(), DType::BF16, {count})};
    const std::array<Tensor, 2> staging{
        Tensor(fixture.storage[0]->staging.data(), DType::BF16, {count}),
        Tensor(fixture.storage[1]->staging.data(), DType::BF16, {count})};
    ops::allreduce_sum(input, staging, fixture.execution, transfer);
}

void gather(PeerTransferFixture& fixture, const ops::PeerTransfer& transfer) {
    const std::array<Tensor, 2> destination{
        Tensor(fixture.storage[0]->gathered.data(), DType::BF16, {1, gathered_elements}),
        Tensor(fixture.storage[1]->gathered.data(), DType::BF16, {1, gathered_elements})};
    const std::array<Tensor, 2> part{
        Tensor(fixture.storage[0]->input.data(), DType::BF16, {1, gather_rows[0]}),
        Tensor(fixture.storage[1]->input.data(), DType::BF16, {1, gather_rows[1]})};
    ops::allgather_rows(destination, part, fixture.execution, transfer);
}

void snapshot(PeerTransferFixture& fixture, const ops::PeerTransfer& transfer, int call,
              bool is_gather = false, bool publication = true) {
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        const auto stream = fixture.execution.dev[rank]->stream;
        auto* output = static_cast<std::uint16_t*>(fixture.storage[rank]->sum_history.data()) +
                       static_cast<std::size_t>(call) * elements;
        // Retain the entire input, also exposing writes beyond a partial sum's admitted shape.
        CUDA_CHECK(cudaMemcpyAsync(output, fixture.storage[rank]->input.data(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        if (publication) {
            auto* published = static_cast<std::uint16_t*>(fixture.storage[rank]->host_source_history.data()) +
                              static_cast<std::size_t>(call) * elements;
            CUDA_CHECK(cudaMemcpyAsync(published, transfer.host_buffer(rank), bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (is_gather) {
            auto* gathered = static_cast<std::uint16_t*>(fixture.storage[rank]->gather_history.data()) +
                             static_cast<std::size_t>(call) * gathered_elements;
            CUDA_CHECK(cudaMemcpyAsync(gathered, fixture.storage[rank]->gathered.data(),
                                       static_cast<std::size_t>(gathered_elements) * 2,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }
}

int guards(PeerTransferFixture& fixture, bool check_host = true, bool untouched_host = false) {
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        for (const auto* buffer : {&fixture.storage[rank]->input, &fixture.storage[rank]->staging,
                                  &fixture.storage[rank]->gathered, &fixture.storage[rank]->sum_history,
                                  &fixture.storage[rank]->host_source_history,
                                  &fixture.storage[rank]->gather_history, &fixture.storage[rank]->skew}) {
            failures += buffer->verify_guards("pipeline integration");
        }
        if (check_host) {
            const auto* host = static_cast<const unsigned char*>(fixture.transfer.host_buffer(rank));
            for (std::size_t i = untouched_host ? 0 : bytes;
                 i < fixture.transfer.host_capacity_bytes(); ++i) {
                if (host[i] != PeerTransferFixture::host_canary) {
                    std::cerr << "pipeline pinned guard/fallback changed at " << i << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

int verify_sum(PeerTransferFixture& fixture, int call, int count) {
    // Independent FP64 sum from represented operands. Their bounded dyadic sum is exactly
    // FP32-representable; the only rounding is the mathematical result's final BF16 storage.
    auto expected_sum = fixture.sum_oracle(call);
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        std::vector<std::uint16_t> expected(source(fixture, rank, call),
                                            source(fixture, rank, call) + elements);
        std::copy_n(expected_sum.begin(), count, expected.begin());
        const auto* actual = static_cast<const std::uint16_t*>(fixture.storage[rank]->sum_history.data()) +
                             static_cast<std::size_t>(call) * elements;
        failures += verify_exact("every sum and untouched input tail",
                                  from_device<std::uint16_t>(actual, elements), expected);
    }
    return failures;
}

int verify_sequence(PeerTransferFixture& fixture) {
    int failures = guards(fixture);
    std::array<std::vector<std::uint16_t>, 2> published;
    for (int call = 0; call < calls; ++call) {
        const bool is_gather = call == 3;
        const int count = call == 1 ? partial_elements : elements;
        if (!is_gather) { failures += verify_sum(fixture, call, count); }
        for (int rank = 0; rank < 2; ++rank) {
            const auto* own_source = source(fixture, is_gather ? 1 - rank : rank, call);
            if (published[rank].empty()) { published[rank].resize(elements); }
            // Partial routes overwrite only their prefix of the SAME pipeline host buffer.
            std::copy_n(own_source, is_gather ? gather_rows[rank] : count, published[rank].begin());
            fixture.select(rank);
            const auto* actual = static_cast<const std::uint16_t*>(fixture.storage[rank]->host_source_history.data()) +
                                 static_cast<std::size_t>(call) * elements;
            failures += verify_exact("every shared pinned publication and retained tail",
                from_device<std::uint16_t>(actual, elements), published[rank]);
            if (is_gather) {
                const auto* input = static_cast<const std::uint16_t*>(fixture.storage[rank]->sum_history.data()) +
                                    static_cast<std::size_t>(call) * elements;
                failures += verify_exact("gather input remains unchanged",
                    from_device<std::uint16_t>(input, elements),
                    std::vector<std::uint16_t>(own_source, own_source + elements));
                std::vector<std::uint16_t> expected;
                for (int part = 0; part < 2; ++part) {
                    const auto* peer_source = source(fixture, 1 - part, call);
                    expected.insert(expected.end(), peer_source, peer_source + gather_rows[part]);
                }
                const auto* gathered = static_cast<const std::uint16_t*>(fixture.storage[rank]->gather_history.data()) +
                                       static_cast<std::size_t>(call) * gathered_elements;
                failures += verify_exact("unequal exact gather including special BF16 bytes",
                    from_device<std::uint16_t>(gathered, gathered_elements), expected);
            }
        }
    }
    return failures;
}

int eager_case(const std::vector<int>& devices, Schedule schedule) {
    PeerTransferFixture fixture(devices, elements, calls, gather_rows, bytes + guard_bytes);
    if (schedule == Schedule::WholeBuffer) {
        fixture.transfer = ops::PeerTransfer(fixture.execution, bytes + guard_bytes, schedule);
    }
    const bool automatic = schedule == Schedule::Automatic;
    require(fixture.transfer.uses_two_tile_allreduce(bytes) == automatic,
            "qualified Automatic pipeline or WholeBuffer control route unavailable");
    require(!fixture.transfer.uses_two_tile_allreduce(partial_elements * 2u),
            "partial sum unexpectedly admitted to the exact 10 MiB pipeline");
    require(fixture.transfer.uses_host_staging({partial_elements * 2u, partial_elements * 2u}) &&
            fixture.transfer.uses_host_staging({gather_rows[0] * 2u, gather_rows[1] * 2u}),
            "partial routes must actually share explicit pinned staging");
    int failures = 0;
    for (int epoch = 0; epoch < 3; ++epoch) {
        prepare(fixture, epoch);
        for (int call = 0; call < calls; ++call) {
            fixture.skew((call + epoch) % 2);
            fixture.upload(call, call == 3);
            if (call == 3) { gather(fixture, fixture.transfer); }
            else { sum(fixture, fixture.transfer, call == 1 ? partial_elements : elements); }
            snapshot(fixture, fixture.transfer, call, call == 3);
        }
        fixture.retire(); // no host synchronization inside the changing five-call sequence
        failures += verify_sequence(fixture);
    }
    std::cout << "shared-buffer chain devices=" << devices[0] << ',' << devices[1]
              << " schedule=" << (automatic ? "Automatic" : "WholeBuffer")
              << " failures=" << failures << '\n';
    return failures;
}

int captured_case(const std::vector<int>& devices) {
    PeerTransferFixture fixture(devices, elements, calls, gather_rows, bytes + guard_bytes);
    DecodeGraphPeerBridge bridge(devices[0], devices[1]);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    require(fixture.transfer.uses_two_tile_allreduce(bytes), "capture setup lacks eager pipeline");
    prepare(fixture, 17);
    fixture.select(0);
    definition.capture(fixture.execution.dev[0]->stream, [&] {
        require(!fixture.transfer.uses_two_tile_allreduce(bytes) &&
                    !fixture.transfer.uses_host_staging({bytes, bytes}),
                "capture selected eager pipeline/pinned storage");
        for (int call = 0; call < 2; ++call) {
            fixture.upload(call);
            sum(fixture, fixture.transfer, elements);
            snapshot(fixture, fixture.transfer, call, false, false);
        }
    }, DecodeGraphPeerCapture{&bridge, fixture.execution.dev[1]->stream});
    executable.instantiate(definition);
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        prepare(fixture, 17 + replay);
        fixture.skew(replay);
        fixture.select(0);
        bridge.gate_launch(fixture.execution.dev[1]->stream, fixture.execution.dev[0]->stream);
        executable.launch(fixture.execution.dev[0]->stream);
        fixture.retire();
        failures += verify_sum(fixture, 0, elements) + verify_sum(fixture, 1, elements);
        failures += guards(fixture, true, true); // every pinned byte untouched proves fallback
        require(fixture.transfer.uses_two_tile_allreduce(bytes), "capture disabled later eager pipeline");
        fixture.upload(2);
        sum(fixture, fixture.transfer, elements);
        snapshot(fixture, fixture.transfer, 2);
        fixture.retire();
        failures += verify_sum(fixture, 2, elements) + guards(fixture);
    }
    std::cout << "captured 10 MiB fallback devices=" << devices[0] << ',' << devices[1]
              << " failures=" << failures << '\n';
    return failures;
}

int moved_pending_owner_case(const std::vector<int>& devices) {
    PeerTransferFixture fixture(devices, elements, calls, gather_rows, bytes + guard_bytes);
    prepare(fixture, 29);
    {
        // Construct the assignment destination BEFORE issuing work, so no later allocation is
        // mistaken for the in-flight move-constructor's lifetime protection.
        ops::PeerTransfer destination(fixture.execution, bytes + guard_bytes, Schedule::WholeBuffer);
        fixture.skew(0);
        fixture.upload(0);
        sum(fixture, fixture.transfer, elements);
        snapshot(fixture, fixture.transfer, 0);
        void* original = fixture.transfer.host_buffer(0);
        ops::PeerTransfer moved(std::move(fixture.transfer));
        require(!fixture.transfer.live() && moved.host_buffer(0) == original &&
                    moved.uses_two_tile_allreduce(bytes), "move lost the active transfer resource");
        fixture.upload(1);
        sum(fixture, moved, elements);
        snapshot(fixture, moved, 1);
        void* former_destination = destination.host_buffer(0);
        destination = std::move(moved);
        // Assignment swaps complete owners; its source receives the old WholeBuffer owner.
        require(destination.host_buffer(0) == original && destination.uses_two_tile_allreduce(bytes) &&
                    moved.live() && moved.host_buffer(0) == former_destination &&
                    !moved.uses_two_tile_allreduce(bytes), "move assignment lost/swapped partial ownership");
        fixture.skew(1);
        fixture.upload(2);
        sum(fixture, destination, elements);
        snapshot(fixture, destination, 2);
        // No explicit wait. Resource teardown must retire both main and auxiliary streams.
    }
    int failures = guards(fixture, false);
    for (int call = 0; call < 3; ++call) {
        failures += verify_sum(fixture, call, elements);
        for (int rank = 0; rank < 2; ++rank) {
            fixture.select(rank);
            const auto* published = static_cast<const std::uint16_t*>(fixture.storage[rank]->host_source_history.data()) +
                                    static_cast<std::size_t>(call) * elements;
            failures += verify_exact("moved owner retains every pinned publication",
                from_device<std::uint16_t>(published, elements),
                std::vector<std::uint16_t>(source(fixture, rank, call), source(fixture, rank, call) + elements));
        }
    }
    std::cout << "in-flight moved owner devices=" << devices[0] << ',' << devices[1]
              << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main() {
    try {
#ifndef _WIN32
        std::cout << "SKIP: two-tile production profile is qualified on Windows\n";
        return 77;
#else
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count < 2) { std::cout << "SKIP: two CUDA devices required\n"; return 77; }
        for (int device = 0; device < 2; ++device) {
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            if (prop.major != 12 || prop.minor != 0 || prop.multiProcessorCount != 70) {
                std::cout << "SKIP: two sm_120 70-SM devices required\n"; return 77;
            }
            int peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&peer, device, 1 - device));
            if (peer) { std::cout << "SKIP: qualified pair has no P2P in either direction\n"; return 77; }
        }
        int failures = 0;
        for (const std::vector<int> devices : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
            failures += eager_case(devices, Schedule::Automatic);
            failures += eager_case(devices, Schedule::WholeBuffer);
            failures += captured_case(devices);
            failures += moved_pending_owner_case(devices);
        }
        std::cout << (failures ? "FAIL" : "OK") << " production two-tile integration\n";
        return failures ? 1 : 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "FAIL production two-tile integration: " << error.what() << '\n';
        return 1;
    }
}
