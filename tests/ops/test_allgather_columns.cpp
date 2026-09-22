#include "ops/allgather_columns_fixture.h"

#include "core/layout.h"

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

// Match the model head's composition: an existing arena prefix, the live local logit shard,
// then the Op's scoped peer scratch. Odd shard extents force alignment at the actual cursor.
int arena_composition_case(const std::vector<int>& devices, int rows0, int rows1, int columns) {
    struct ComposedStorage {
        int device;
        cudaStream_t stream;
        GuardedDeviceBuffer backing;
        WorkspaceArena arena;

        ComposedStorage(int device_id, cudaStream_t owner_stream, std::size_t capacity)
            : device(device_id), stream(owner_stream), backing(capacity),
              arena(DeviceSpan{backing.data(), backing.bytes()}) {}
        ~ComposedStorage() {
            (void)cudaSetDevice(device);
            (void)cudaStreamSynchronize(stream);
        }
    };
    constexpr std::size_t prefix_bytes = 37;
    constexpr std::uint8_t canary = 0x6b;
    AllgatherColumnsFixture fixture(devices, rows0, rows1, columns, 1, true);
    fixture.prepare(71);
    std::array<std::unique_ptr<ComposedStorage>, 2> storage;
    std::array<WorkspaceArena*, 2> workspace;
    std::array<Tensor, 2> part;
    std::array<Tensor, 2> destination;
    std::array<std::size_t, 2> part_offset;
    std::array<std::size_t, 2> before;
    std::array<std::size_t, 2> peak;
    for (int rank = 0; rank < 2; ++rank) {
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(prefix_bytes);
        (void)layout.alloc(DType::BF16, {fixture.rows[rank], columns});
        const auto scratch = ops::allgather_columns_workspace_capacity_bytes(
            fixture.rows[1 - rank], columns);
        if (scratch != 0) {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(scratch);
        }
        peak[rank] = layout.peak_bytes(1);
        fixture.select(rank);
        storage[rank] = std::make_unique<ComposedStorage>(
            fixture.execution.dev[rank]->device, fixture.execution.dev[rank]->stream, peak[rank]);
        auto& item = *storage[rank];
        item.backing.fill(canary);
        CUDA_CHECK(cudaDeviceSynchronize()); // retire legacy-stream guard initialization
        (void)item.arena.alloc_bytes(prefix_bytes);
        part[rank] = item.arena.alloc(DType::BF16, {fixture.rows[rank], columns});
        part_offset[rank] = static_cast<std::uint8_t*>(part[rank].data) -
                            static_cast<std::uint8_t*>(item.backing.data());
        before[rank] = item.arena.used();
        workspace[rank] = &item.arena;
        destination[rank] = Tensor(fixture.storage[rank]->output.data(), DType::BF16,
                                    {fixture.total_rows, columns});
        CUDA_CHECK(cudaMemcpyAsync(part[rank].data, fixture.host_part(rank, 0), part[rank].bytes(),
                                   cudaMemcpyHostToDevice, fixture.execution.dev[rank]->stream));
    }
    fixture.poison_outputs();
    ops::allgather_columns(destination, part, workspace, fixture.execution, fixture.transfer);
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto& arena = storage[rank]->arena;
        if (arena.used() != before[rank] || arena.peak_used() != peak[rank]) {
            std::cerr << "composed gather arena rank " << rank << ": cursor=" << arena.used()
                      << " expected=" << before[rank] << " peak=" << arena.peak_used()
                      << " planned=" << peak[rank] << '\n';
            ++failures;
        }
    }
    fixture.retire();
    const auto expected = fixture.oracle(0);
    for (int rank = 0; rank < 2; ++rank) {
        fixture.select(rank);
        const auto& item = *storage[rank];
        failures += verify_exact("composed gather output bits",
            from_device<std::uint16_t>(destination[rank].data, expected.size()), expected);
        const std::size_t count = static_cast<std::size_t>(fixture.rows[rank]) * columns;
        const auto* original = fixture.host_part(rank, 0);
        failures += verify_exact("composed gather live input unchanged",
            from_device<std::uint16_t>(part[rank].data, count),
            std::vector<std::uint16_t>(original, original + count));
        failures += verify_exact("composed gather prefix and leading alignment gap unchanged",
            from_device<std::uint8_t>(item.backing.data(), part_offset[rank]),
            std::vector<std::uint8_t>(part_offset[rank], canary));
        if (columns > 1) {
            const std::size_t gap = (256 - before[rank] % 256) % 256;
            const auto* gap_start = static_cast<const std::uint8_t*>(item.backing.data()) + before[rank];
            failures += verify_exact("composed gather peer alignment gap unchanged",
                from_device<std::uint8_t>(gap_start, gap), std::vector<std::uint8_t>(gap, canary));
        }
        failures += item.backing.verify_guards("composed gather arena guards");
        failures += fixture.storage[rank]->output.verify_guards("composed gather output guards");
    }
    std::cout << (failures ? "FAIL" : "OK") << " same-arena packed gather rows=" << rows0
              << '+' << rows1 << " T=" << columns << '\n';
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
            if (route == ColumnGatherRoute::Packed) {
                failures += arena_composition_case(devices, shape[0], shape[1], shape[2]);
            }
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
