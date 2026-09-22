#include "ops/allgather_columns_fixture.h"

#include <iostream>
#include <string_view>

using namespace ninfer;
using namespace ninfer::test;

namespace {

void enqueue_case(AllgatherColumnsFixture& fixture, ColumnGatherRoute route) {
    for (int call = 0; call < fixture.calls; ++call) {
        if (call == 0 || call == fixture.calls / 2) { fixture.skew(call % 2); }
        fixture.issue(route, call);
    }
}

int eager_case(const std::vector<int>& devices, int rows0, int rows1, int columns,
                ColumnGatherRoute route, bool provision_host) {
    AllgatherColumnsFixture fixture(devices, rows0, rows1, columns, 3, provision_host);
    int failures = 0;
    for (int round = 0; round < 2; ++round) {
        fixture.prepare(13 + round);
        fixture.reset_host_canaries();
        const bool staged = fixture.eager_staged(route);
        // Assert the exercised route independently of the production selector. In these
        // fixtures the provisioned large shards exceed the 64 KiB eager threshold; the
        // small edge case and unprovisioned owners must retain the UVA copy route.
        const int transferred_columns = route == ColumnGatherRoute::Baseline ? 1 : columns;
        const bool expected_staged = provision_host &&
            static_cast<std::size_t>(std::max(rows0, rows1)) * transferred_columns * 2 >=
                64u * 1024u;
        if (staged != expected_staged) {
            throw std::runtime_error("gather fixture did not select its expected eager transport");
        }
        enqueue_case(fixture, route);
        fixture.retire();
        failures += fixture.verify(true);
        failures += fixture.verify_transport(route, staged, fixture.calls - 1);
    }
    std::cout << (failures ? "FAIL" : "OK") << " eager route=" << column_gather_name(route)
              << " rows=" << rows0 << '+' << rows1 << " T=" << columns
              << " pinned_capacity=" << fixture.transfer.host_capacity_bytes()
              << " direct_p2p=" << fixture.direct_p2p << '\n';
    return failures;
}

// The second definition has identical logical topology but distinct live device/host addresses,
// events and streams on the SAME device contexts. This exercises whole-graph update, which is
// the engine's API, without assuming one executable can change token-width topology.
int graph_update_case(const std::vector<int>& devices, int rows0, int rows1, int columns,
                       ColumnGatherRoute route) {
    AllgatherColumnsFixture first(devices, rows0, rows1, columns, 3, true);
    AllgatherColumnsFixture second(devices, rows0, rows1, columns, 3, true);
    DecodeGraphPeerBridge first_bridge(devices[0], devices[1]);
    DecodeGraphPeerBridge second_bridge(devices[0], devices[1]);
    DecodeGraphDefinition first_definition;
    DecodeGraphDefinition second_definition;
    DecodeGraphExecutable executable;
    const auto capture = [&](AllgatherColumnsFixture& fixture, DecodeGraphPeerBridge& bridge,
                              DecodeGraphDefinition& definition) {
        fixture.prepare(19);
        fixture.select(0);
        definition.capture(fixture.execution.dev[0]->stream,
                            [&] { enqueue_case(fixture, route); },
                            {&bridge, fixture.execution.dev[1]->stream});
    };
    capture(first, first_bridge, first_definition);
    capture(second, second_bridge, second_definition);
    first.select(0);
    executable.instantiate(first_definition);
    int failures = 0;
    const auto replay = [&](AllgatherColumnsFixture& fixture, DecodeGraphPeerBridge& bridge,
                             int epoch) {
        fixture.prepare(epoch);
        fixture.reset_host_canaries();
        fixture.skew(1);
        fixture.select(0);
        bridge.gate_launch(fixture.execution.dev[1]->stream, fixture.execution.dev[0]->stream);
        executable.launch(fixture.execution.dev[0]->stream);
        fixture.retire();
        return fixture.verify(true) + fixture.verify_transport(route, false, fixture.calls - 1);
    };
    failures += replay(first, first_bridge, 31);
    failures += replay(first, first_bridge, 32);
    first.select(0);
    executable.update(second_definition);
    failures += replay(second, second_bridge, 47);
    second.select(0);
    executable.update(first_definition);
    failures += replay(first, first_bridge, 59);
    std::cout << (failures ? "FAIL" : "OK") << " capture/replay/update route="
              << column_gather_name(route) << " rows=" << rows0 << '+' << rows1
              << " T=" << columns << '\n';
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<int> devices{0, 1};
        ColumnGatherRoute route = ColumnGatherRoute::Packed;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--reverse-devices") { devices = {1, 0}; }
            else if (argument == "--candidate-direct2d") { route = ColumnGatherRoute::Direct2D; }
            else if (argument == "--baseline") { route = ColumnGatherRoute::Baseline; }
            else {
                throw std::invalid_argument("usage: ninfer_allgather_columns_test "
                    "[--reverse-devices] [--candidate-direct2d | --baseline]");
            }
        }
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count < 2) { std::cout << "SKIP: two CUDA devices required\n"; return 77; }
        int failures = 0;
        for (const auto& shape : std::array<std::array<int, 3>, 4>{
                 {{124160, 124160, 1}, {124160, 124160, 4},
                  {124159, 124161, 4}, {23, 19, 3}}}) {
            for (bool pinned : {false, true}) {
                failures += eager_case(devices, shape[0], shape[1], shape[2], route, pinned);
            }
            failures += graph_update_case(devices, shape[0], shape[1], shape[2], route);
        }
        std::cout << (failures ? "FAIL" : "OK") << " allgather columns devices="
                  << devices[0] << ',' << devices[1] << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        // An unsupported direct-2D experiment is reported explicitly with failure status;
        // it is never silently relabeled as a successfully qualified packed Op.
        std::cerr << "allgather columns: " << error.what() << '\n';
        return 1;
    }
}
