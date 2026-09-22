#include "ops/allgather_columns_fixture.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Options {
    int columns = 4;
    int calls = 1;
    int samples = 31;
    int warmup = 10;
    bool reverse = false;
    bool direct2d = false;
    bool host_staging = true;
    std::string mode = "both";
};

Options parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--reverse-devices") { out.reverse = true; continue; }
        if (argument == "--with-direct2d") { out.direct2d = true; continue; }
        if (argument == "--implicit-only") { out.host_staging = false; continue; }
        if (argument == "--help" || argument == "-h") {
            std::cout << "usage: ninfer_allgather_columns_bench [--cols 1|4] [--calls N] "
                         "[--samples N] [--warmup N] [--mode eager|graph|both] "
                         "[--with-direct2d] [--implicit-only] [--reverse-devices]\n";
            std::exit(0);
        }
        if (++i >= argc) { throw std::invalid_argument("missing option value"); }
        const std::string value(argv[i]);
        if (argument == "--mode") { out.mode = value; continue; }
        std::size_t used = 0;
        const int number = std::stoi(value, &used);
        if (used != value.size()) { throw std::invalid_argument("invalid integer value"); }
        if (argument == "--cols") { out.columns = number; }
        else if (argument == "--calls") { out.calls = number; }
        else if (argument == "--samples") { out.samples = number; }
        else if (argument == "--warmup") { out.warmup = number; }
        else { throw std::invalid_argument("unknown option"); }
    }
    if ((out.columns != 1 && out.columns != 4) || out.calls < 1 || out.samples < 3 ||
        out.warmup < 1 || (out.mode != "eager" && out.mode != "graph" && out.mode != "both")) {
        throw std::invalid_argument("cols must be 1 or 4, counts positive, samples>=3, mode eager|graph|both");
    }
    return out;
}

struct Captured {
    DecodeGraphPeerBridge bridge;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    Captured(AllgatherColumnsFixture& fixture, ColumnGatherRoute route, int calls)
        : bridge(fixture.execution.dev[0]->device, fixture.execution.dev[1]->device) {
        fixture.select(0);
        definition.capture(fixture.execution.dev[0]->stream, [&] {
            for (int call = 0; call < calls; ++call) { fixture.gather(route); }
        }, {&bridge, fixture.execution.dev[1]->stream});
        executable.instantiate(definition);
        executable.upload(fixture.execution.dev[0]->stream);
        fixture.retire();
    }
};

struct Timers {
    const ExecutionContext& execution;
    cudaEvent_t begin = nullptr;
    cudaEvent_t peer_done = nullptr;
    cudaEvent_t complete = nullptr;
    explicit Timers(const ExecutionContext& owner) : execution(owner) {
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&complete));
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        CUDA_CHECK(cudaEventCreateWithFlags(&peer_done, cudaEventDisableTiming));
    }
    ~Timers() {
        (void)cudaSetDevice(execution.dev[1]->device);
        (void)cudaEventDestroy(peer_done);
        (void)cudaSetDevice(execution.dev[0]->device);
        (void)cudaEventDestroy(complete);
        (void)cudaEventDestroy(begin);
    }
    void start() {
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        CUDA_CHECK(cudaEventRecord(begin, execution.dev[0]->stream));
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        CUDA_CHECK(cudaStreamWaitEvent(execution.dev[1]->stream, begin, 0));
    }
    float finish() {
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        CUDA_CHECK(cudaEventRecord(peer_done, execution.dev[1]->stream));
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        CUDA_CHECK(cudaStreamWaitEvent(execution.dev[0]->stream, peer_done, 0));
        CUDA_CHECK(cudaEventRecord(complete, execution.dev[0]->stream));
        CUDA_CHECK(cudaEventSynchronize(complete));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, complete));
        return milliseconds;
    }
};

struct Measurement { double enqueue; double wall; double device; };

Measurement measure(AllgatherColumnsFixture& fixture, ColumnGatherRoute route,
                      Captured* captured, Timers& timers, int calls) {
    if (captured) {
        fixture.select(0);
        captured->bridge.gate_launch(fixture.execution.dev[1]->stream,
                                     fixture.execution.dev[0]->stream);
    }
    timers.start();
    const auto begin = std::chrono::steady_clock::now();
    if (captured) {
        fixture.select(0);
        captured->executable.launch(fixture.execution.dev[0]->stream);
    } else {
        for (int call = 0; call < calls; ++call) { fixture.gather(route); }
    }
    const auto enqueued = std::chrono::steady_clock::now();
    // For graphs the origin stream also contains the capture's peer join. For eager calls the
    // explicit event join covers both streams. No timings subtract events across devices.
    const float device_ms = timers.finish();
    const auto completed = std::chrono::steady_clock::now();
    return {std::chrono::duration<double, std::micro>(enqueued - begin).count() / calls,
            std::chrono::duration<double, std::micro>(completed - begin).count() / calls,
            static_cast<double>(device_ms) * 1000.0 / calls};
}

void report(const char* mode, ColumnGatherRoute route, const std::vector<Measurement>& samples) {
    std::array<std::vector<double>, 3> values;
    for (const auto& sample : samples) {
        values[0].push_back(sample.enqueue);
        values[1].push_back(sample.wall);
        values[2].push_back(sample.device);
    }
    for (auto& item : values) { std::sort(item.begin(), item.end()); }
    const std::size_t median = samples.size() / 2;
    const std::size_t p95 = (samples.size() * 95 + 99) / 100 - 1;
    std::cout << std::setprecision(9)
              << "{\"type\":\"summary\",\"mode\":\"" << mode << "\",\"route\":\""
              << column_gather_name(route) << "\",\"samples\":" << samples.size()
              << ",\"median_enqueue_us\":" << values[0][median]
              << ",\"median_complete_wall_us\":" << values[1][median]
              << ",\"median_joined_device_us\":" << values[2][median]
              << ",\"p95_complete_wall_us\":" << values[1][p95]
              << ",\"p95_joined_device_us\":" << values[2][p95] << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        if (cuda_unavailable()) { std::cout << "SKIP: no CUDA device\n"; return 77; }
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count < 2) { std::cout << "SKIP: two CUDA devices required\n"; return 77; }
        const std::vector<int> devices = options.reverse ? std::vector<int>{1, 0}
                                                         : std::vector<int>{0, 1};
        AllgatherColumnsFixture fixture(devices, 124160, 124160, options.columns, 1,
                                         options.host_staging);
        fixture.prepare(23);
        fixture.upload(0);
        fixture.retire();
        Timers timers(fixture.execution);
        std::vector<ColumnGatherRoute> routes{ColumnGatherRoute::Baseline, ColumnGatherRoute::Packed};
        if (options.direct2d) { routes.push_back(ColumnGatherRoute::Direct2D); }
        // Both admitted shapes exceed the eager staging threshold even in the per-column
        // baseline. Verify the requested transport before labeling any measurements.
        for (const auto route : routes) {
            if (fixture.eager_staged(route) != options.host_staging) {
                throw std::runtime_error("benchmark did not select its expected eager transport");
            }
        }
        const char* eager_transport = options.host_staging ? "pinned-staging" : "uva-copy";
        std::cout << "{\"type\":\"configuration\",\"scope\":\"exact-gather-only\","
                     "\"rows_per_rank\":124160,\"cols\":" << options.columns
                  << ",\"calls_per_sample\":" << options.calls
                  << ",\"warmup\":" << options.warmup
                  << ",\"samples\":" << options.samples
                  << ",\"devices\":[" << devices[0] << ',' << devices[1] << ']'
                  << ",\"direct_p2p\":" << (fixture.direct_p2p ? "true" : "false")
                  << ",\"eager_transport\":\"" << eager_transport << '\"'
                  << ",\"pinned_capacity_per_rank\":" << fixture.transfer.host_capacity_bytes()
                  << ",\"packed_scratch_per_rank\":"
                  << ops::allgather_columns_workspace_capacity_bytes(124160, options.columns)
                  << ",\"order\":\"rotating paired\",\"input\":\"resident raw BF16 bits\"}\n";
        for (const std::string mode : {std::string("eager"), std::string("graph")}) {
            if (options.mode != "both" && options.mode != mode) { continue; }
            std::vector<std::unique_ptr<Captured>> graphs(routes.size());
            if (mode == "graph") {
                for (std::size_t route = 0; route < routes.size(); ++route) {
                    graphs[route] = std::make_unique<Captured>(fixture, routes[route], options.calls);
                }
            }
            // Qualification is outside timing and occurs independently for every candidate.
            // Poisoning prevents a no-op candidate from inheriting the previous route's output.
            const auto qualify_routes = [&] {
                for (std::size_t route = 0; route < routes.size(); ++route) {
                    fixture.poison_outputs();
                    fixture.retire();
                    fixture.reset_host_canaries();
                    (void)measure(fixture, routes[route], graphs[route].get(), timers, options.calls);
                    const bool staged = mode == "eager" && options.host_staging;
                    if (fixture.verify(false) + fixture.verify_transport(routes[route], staged, 0) != 0) {
                        throw std::runtime_error("gather benchmark exact oracle/transport failed");
                    }
                }
            };
            qualify_routes();
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                for (std::size_t item = 0; item < routes.size(); ++item) {
                    const std::size_t route = (iteration + item) % routes.size();
                    (void)measure(fixture, routes[route], graphs[route].get(), timers, options.calls);
                }
            }
            std::vector<std::vector<Measurement>> samples(routes.size());
            for (int iteration = 0; iteration < options.samples; ++iteration) {
                for (std::size_t item = 0; item < routes.size(); ++item) {
                    const std::size_t route = (iteration + item) % routes.size();
                    const Measurement sample = measure(fixture, routes[route], graphs[route].get(),
                                                         timers, options.calls);
                    samples[route].push_back(sample);
                    std::cout << std::setprecision(9) << "{\"type\":\"sample\",\"mode\":\""
                              << mode << "\",\"route\":\"" << column_gather_name(routes[route])
                              << "\",\"transport\":\""
                              << (mode == "eager" ? eager_transport : "uva-copy")
                              << "\",\"pair\":" << iteration
                              << ",\"enqueue_us\":" << sample.enqueue
                              << ",\"complete_wall_us\":" << sample.wall
                              << ",\"joined_device_us\":" << sample.device << "}\n";
                }
            }
            // Check the final timed result before any new write can hide corruption,
            // then qualify every route again with fresh poison after repeated reuse.
            if (fixture.verify(false) != 0) {
                throw std::runtime_error("gather benchmark final timed output/input/guard failed");
            }
            qualify_routes();
            for (std::size_t route = 0; route < routes.size(); ++route) {
                report(mode.c_str(), routes[route], samples[route]);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "allgather columns benchmark: " << error.what() << '\n';
        return 1;
    }
}
