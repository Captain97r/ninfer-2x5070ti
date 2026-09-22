#include "ops/argmax_row_parallel_fixture.h"

#include <iostream>
#include <string_view>

using namespace ninfer;
using namespace ninfer::test;

namespace {

enum class Mode { Eager, BaselineGraph, StagedGraph, MailboxGraph, ExhaustedGraph, SmallMailboxGraph };

int run_case(const std::vector<int>& devices, int rows0, int rows1, int valid_rows,
              int columns, Mode mode, bool remap) {
    ArgmaxMailboxEnvironment environment(mode == Mode::StagedGraph ? "0" : "1");
    ArgmaxRowParallelFixture fixture(devices, rows0, rows1, valid_rows, columns, 3);
    const bool captured = mode != Mode::Eager;
    const bool baseline = mode == Mode::BaselineGraph;
    const bool installed = mode != Mode::Eager && mode != Mode::BaselineGraph;
    const int slots = mode == Mode::ExhaustedGraph ? 1 : fixture.calls;
    const std::size_t slot_bytes = mode == Mode::SmallMailboxGraph ? 16 : 65536;
    std::unique_ptr<ops::PeerMailbox> mailbox;
    if (installed) { mailbox = std::make_unique<ops::PeerMailbox>(fixture.execution, slot_bytes, slots); }
    fixture.prepare(0, true);
    DecodeGraphPeerBridge bridge(devices[0], devices[1]);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    auto body = [&] {
        for (int call = 0; call < fixture.calls; ++call) {
            fixture.issue(baseline, call, true, remap);
        }
    };
    if (captured) {
        fixture.select(0);
        definition.capture(fixture.execution.dev[0]->stream, body,
                            {&bridge, fixture.execution.dev[1]->stream});
        executable.instantiate(definition);
    }
    int failures = 0;
    for (int replay = 0; replay < 4; ++replay) {
        fixture.prepare(replay, true);
        if (mailbox) { ops::PeerMailbox::reset_host_flags(); }
        if (captured) {
            fixture.select(0);
            bridge.gate_launch(fixture.execution.dev[1]->stream, fixture.execution.dev[0]->stream);
            executable.launch(fixture.execution.dev[0]->stream);
        } else {
            body();
        }
        fixture.retire();
        if (mailbox) { mailbox->validate_completed_round(); }
        failures += fixture.verify(remap);
    }
    std::cout << (failures ? "FAIL" : "OK") << " rows=" << rows0 << "+" << rows1
              << " valid=" << valid_rows << " T=" << columns << " mode=" << static_cast<int>(mode)
              << " remap=" << remap << " primary=" << devices[0] << "\n";
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count < 2)) {
        std::cout << "SKIP: row-parallel argmax requires two CUDA devices\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        std::vector<int> devices{0, 1};
        if (argc == 2 && std::string_view(argv[1]) == "--reverse-devices") { devices = {1, 0}; }
        else if (argc != 1) { throw std::invalid_argument("usage: ninfer_argmax_row_parallel_test [--reverse-devices]"); }
        int failures = 0;
        for (Mode mode : {Mode::Eager, Mode::BaselineGraph, Mode::StagedGraph, Mode::MailboxGraph,
                          Mode::ExhaustedGraph}) {
            failures += run_case(devices, 65536, 65536, 131072, 1, mode, false);
            failures += run_case(devices, 65536, 65536, 131072, 8, mode, true);
            failures += run_case(devices, 23, 19, 31, 3, mode, false);
            failures += run_case(devices, 23, 19, 23, 3, mode, false);
            // A one-row remapping table maps every physical row to zero and could hide
            // selection from padding. Check the raw global index for this domain.
            failures += run_case(devices, 23, 19, 1, 3, mode, false);
        }
        // 16-byte requested slots align to 256 bytes. Seventeen candidates deliberately exceed
        // that capacity and exercise staged fallback inside the same public captured call.
        failures += run_case(devices, 23, 19, 31, 17, Mode::SmallMailboxGraph, true);
        std::cout << (failures ? "FAIL" : "OK") << " row-parallel argmax\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "row-parallel argmax: " << error.what() << "\n";
        return 1;
    }
}
