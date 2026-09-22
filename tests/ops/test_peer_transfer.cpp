#include "ops/peer_transfer_fixture.h"

#include <iostream>

using namespace ninfer;
using namespace ninfer::test;

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

int eager_case(const std::vector<int>& devices, int elements, int calls,
               std::size_t capacity, bool expected_route) {
    PeerTransferFixture fixture(devices, elements, calls, {elements, elements / 3}, capacity);
    const std::size_t bytes = static_cast<std::size_t>(elements) * 2;
    require(fixture.transfer.uses_host_staging({bytes, bytes}) == expected_route,
            "unexpected eager transport selection");
    fixture.prepare(3);
    fixture.skew(0);
    for (int call = 0; call < calls; ++call) {
        if (call == calls / 2) { fixture.skew(1); }
        fixture.issue(call);
    }
    fixture.retire();
    const int failures = fixture.verify(expected_route);
    std::cout << "peer transfer eager bytes=" << bytes << " calls=" << calls
              << " devices=" << devices[0] << ',' << devices[1]
              << " route=" << (expected_route ? "explicit-pinned" : "implicit")
              << " failures=" << failures << '\n';
    return failures;
}

int independent_owners_case() {
    constexpr int first_elements = 65537;
    constexpr int second_elements = 65549;
    constexpr int calls = 4;
    PeerTransferFixture first({0, 1}, first_elements, calls,
                               {first_elements, first_elements / 3}, first_elements * 2 + 256);
    PeerTransferFixture second({0, 1}, second_elements, calls,
                                {second_elements / 2, second_elements}, second_elements * 2 + 256);
    require(first.transfer.host_buffer(0) != second.transfer.host_buffer(0) &&
                first.transfer.host_buffer(1) != second.transfer.host_buffer(1),
            "separate owners share pinned memory");
    bool wrong_owner_rejected = false;
    try { first.sum(second.transfer); }
    catch (const std::invalid_argument&) { wrong_owner_rejected = true; }
    require(wrong_owner_rejected, "collective accepted another owner's stream-bound transfer");

    first.prepare(17);
    second.prepare(23);
    first.skew(0);
    second.skew(1);
    for (int call = 0; call < calls; ++call) {
        first.issue(call);
        second.issue(call);
    }
    first.retire();
    second.retire();
    return first.verify(true) + second.verify(true);
}

int captured_fallback_case() {
    constexpr int elements = 65537;
    constexpr int calls = 3;
    constexpr std::size_t bytes = static_cast<std::size_t>(elements) * 2;
    PeerTransferFixture fixture({0, 1}, elements, calls,
                                 {elements, elements / 3}, bytes + 256);
    DecodeGraphPeerBridge bridge(fixture.execution.dev[0]->device,
                                  fixture.execution.dev[1]->device);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    require(fixture.transfer.uses_host_staging({bytes, bytes}),
            "capture fixture must be eligible before capture");
    fixture.prepare(31);
    fixture.select(0);
    definition.capture(fixture.execution.dev[0]->stream, [&] {
        require(!fixture.transfer.uses_host_staging({bytes, bytes}),
                "explicit pinned eager transport was selected inside capture");
        for (int call = 0; call < calls; ++call) { fixture.issue(call); }
    }, DecodeGraphPeerCapture{&bridge, fixture.execution.dev[1]->stream});
    executable.instantiate(definition);
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        fixture.prepare(31 + replay);
        fixture.skew(1);
        fixture.select(0);
        bridge.gate_launch(fixture.execution.dev[1]->stream, fixture.execution.dev[0]->stream);
        executable.launch(fixture.execution.dev[0]->stream);
        fixture.retire();
        // Exact outputs plus completely untouched pinned buffers prove captured fallback.
        failures += fixture.verify(false);
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices < 2) { std::cout << "SKIP: two CUDA devices required\n"; return 77; }
        int failures = 0;
        // Threshold, odd tail, real 10 MiB [5120,1024] prefill, and reversed rank identities.
        for (const int elements : {32768, 32769, 524288, 5120 * 1024}) {
            failures += eager_case({0, 1}, elements, elements > 524288 ? 2 : 4,
                                    static_cast<std::size_t>(elements) * 2 + 256, true);
        }
        failures += eager_case({1, 0}, 524289, 4, 524289u * 2 + 256, true);
        failures += eager_case({0, 1}, 5120, 4, 5120u * 2 + 256, false);
        failures += eager_case({0, 1}, 65537, 4, 65537u * 2 - 1, false);
        failures += independent_owners_case();
        failures += captured_fallback_case();
        std::cout << (failures ? "FAIL" : "OK") << " explicit peer transfer\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL peer transfer: " << error.what() << '\n';
        return 1;
    }
}
