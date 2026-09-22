#include "ops/argmax_row_parallel_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string_view>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Options {
    int columns = 1;
    int samples = 31;
    int warmup = 10;
    bool reverse = false;
};

Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--reverse-devices") { out.reverse = true; continue; }
        if (arg == "--help" || arg == "-h") {
            std::puts("usage: ninfer_argmax_row_parallel_bench [--cols 1..8] [--samples N] "
                      "[--warmup N] [--reverse-devices]");
            std::exit(0);
        }
        if (i + 1 >= argc) { throw std::invalid_argument("missing benchmark option value"); }
        std::size_t used = 0;
        const std::string value(argv[++i]);
        const int number = std::stoi(value, &used);
        if (used != value.size()) { throw std::invalid_argument("invalid integer option"); }
        if (arg == "--cols") { out.columns = number; }
        else if (arg == "--samples") { out.samples = number; }
        else if (arg == "--warmup") { out.warmup = number; }
        else { throw std::invalid_argument("unknown benchmark option"); }
    }
    if (out.columns < 1 || out.columns > 8 || out.samples < 3 || out.warmup < 1) {
        throw std::invalid_argument("columns must be in [1,8], samples>=3 and warmup>=1");
    }
    return out;
}

struct CapturedSelection {
    const bool uses_mailbox;
    DecodeGraphPeerBridge bridge;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    CapturedSelection(ArgmaxRowParallelFixture& fixture, bool baseline, bool mailbox)
        : uses_mailbox(mailbox),
          bridge(fixture.execution.dev[0]->device, fixture.execution.dev[1]->device) {
        ArgmaxMailboxEnvironment environment(mailbox ? "1" : "0");
        fixture.select(0);
        definition.capture(fixture.execution.dev[0]->stream,
                            [&] { fixture.issue(baseline, 0, false, true); },
                            {&bridge, fixture.execution.dev[1]->stream});
        executable.instantiate(definition);
        executable.upload(fixture.execution.dev[0]->stream);
        fixture.retire();
    }
};

float measure(ArgmaxRowParallelFixture& fixture, CapturedSelection& graph,
               ops::PeerMailbox& mailbox, CudaEventTimer& timer) {
    fixture.select(0);
    mailbox.validate_completed_round();
    ops::PeerMailbox::reset_host_flags();
    graph.bridge.gate_launch(fixture.execution.dev[1]->stream, fixture.execution.dev[0]->stream);
    timer.start();
    graph.executable.launch(fixture.execution.dev[0]->stream);
    timer.record_stop();
    fixture.retire();
    mailbox.validate_completed_round();
    // This fixture owns exactly one slot. A distributed mailbox selection publishes only
    // rank one's flag; both staged routes leave both flags clear after the pre-launch reset.
    // Check after retirement so a silently selected fallback cannot receive a mailbox label.
    const std::uint32_t expected_flag = graph.uses_mailbox ? 1u : 0u;
    if (*mailbox.flag(0, 0) != 0u || *mailbox.flag(1, 0) != expected_flag) {
        throw std::runtime_error("argmax benchmark transport differs from the requested route");
    }
    return timer.elapsed_ms() * 1000.0f;
}

void report(const char* name, std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squares = 0.0;
    for (double value : values) { squares += (value - mean) * (value - mean); }
    const double stddev = std::sqrt(squares / (values.size() - 1));
    const std::size_t p90 = static_cast<std::size_t>(std::ceil(0.9 * values.size())) - 1;
    std::printf("route=%s n=%zu mean_us=%.3f median_us=%.3f p90_us=%.3f stddev_us=%.3f\n",
                name, values.size(), mean, values[values.size() / 2], values[p90], stddev);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count < 2)) {
            std::puts("SKIP: row-parallel argmax benchmark requires two CUDA devices");
            return 77;
        }
        CUDA_CHECK(status);
        ArgmaxMailboxEnvironment environment("1");
        const std::vector<int> devices = options.reverse ? std::vector<int>{1, 0}
                                                         : std::vector<int>{0, 1};
        ArgmaxRowParallelFixture fixture(devices, 65536, 65536, 131072, options.columns, 1);
        fixture.prepare(12, false);
        fixture.upload(0);
        fixture.retire();
        ops::PeerMailbox mailbox(fixture.execution, 40960, 1);
        CapturedSelection baseline(fixture, true, false);
        CapturedSelection distributed(fixture, false, true);
        CapturedSelection staged(fixture, false, false);
        std::array<CapturedSelection*, 3> graphs{&baseline, &distributed, &staged};
        const std::array<const char*, 3> names{"gather_argmax_remap", "distributed_mailbox_remap",
                                             "distributed_staged_remap"};
        fixture.select(0);
        CudaEventTimer timer(*fixture.execution.dev[0]);
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            for (int item = 0; item < 3; ++item) {
                (void)measure(fixture, *graphs[(iteration + item) % 3], mailbox, timer);
            }
        }
        std::array<std::vector<float>, 3> times;
        for (int iteration = 0; iteration < options.samples; ++iteration) {
            for (int item = 0; item < 3; ++item) {
                const int route = (iteration + item) % 3;
                times[route].push_back(measure(fixture, *graphs[route], mailbox, timer));
            }
        }
        // Check the complete selection and remapping result of each measured route against
        // the shared independent oracle. Verification and host transfers are outside timing.
        for (int route = 0; route < 3; ++route) {
            // Each route must write its own answer rather than inherit the preceding route's.
            fixture.select(0);
            CUDA_CHECK(cudaMemsetAsync(fixture.output->data(), 0xcd, fixture.output->bytes(),
                                        fixture.execution.dev[0]->stream));
            fixture.retire();
            (void)measure(fixture, *graphs[route], mailbox, timer);
            if (fixture.verify(true) != 0) { throw std::runtime_error("argmax benchmark oracle failed"); }
        }
        fixture.select(0);
        std::printf("scope=captured_selection_plus_remap rows_per_rank=65536 cols=%d devices=%d,%d "
                    "warmup=%d interleaved_samples=%d logits=resident_bf16 oracle=exact_pass\n",
                    options.columns, devices[0], devices[1], options.warmup, options.samples);
        for (int rank = 0; rank < 2; ++rank) {
            std::printf("rank=%d gpu=%s sms=%d\n", rank, fixture.execution.dev[rank]->props.name,
                        fixture.execution.dev[rank]->props.multiProcessorCount);
        }
        for (int route = 0; route < 3; ++route) { report(names[route], times[route]); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "argmax row-parallel benchmark: %s\n", error.what());
        return 1;
    }
}
