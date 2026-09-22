#include "ops/speculative_rank0_fixture.h"

#include <chrono>
#include <iomanip>
#include <string_view>

using namespace ninfer;
using namespace ninfer::test;
namespace {
struct Options {
    int samples=31, warmup=10;
    bool reverse=false, mailbox=true;
    std::string mode="both", sampling="both";
};
Options parse(int argc,char** argv) {
    Options result;
    for (int i=1; i<argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument=="--reverse-devices") { result.reverse=true; continue; }
        if (argument=="--no-mailbox") { result.mailbox=false; continue; }
        if (argument=="--help" || argument=="-h") {
            std::cout<<"usage: ninfer_speculative_rank0_bench [--samples N] [--warmup N] "
                       "[--mode eager|graph|both] [--sampling greedy|stochastic|both] "
                       "[--no-mailbox] [--reverse-devices]\n";
            std::exit(0);
        }
        if (++i>=argc) { throw std::invalid_argument("missing option value"); }
        const std::string value(argv[i]);
        if (argument=="--mode") { result.mode=value; continue; }
        if (argument=="--sampling") { result.sampling=value; continue; }
        std::size_t used=0; const int number=std::stoi(value,&used);
        if (used!=value.size()) { throw std::invalid_argument("invalid integer option"); }
        if (argument=="--samples") { result.samples=number; }
        else if (argument=="--warmup") { result.warmup=number; }
        else { throw std::invalid_argument("unknown option"); }
    }
    if (result.samples<3 || result.warmup<1 ||
        (result.mode!="eager" && result.mode!="graph" && result.mode!="both") ||
        (result.sampling!="greedy" && result.sampling!="stochastic" && result.sampling!="both")) {
        throw std::invalid_argument("invalid counts, execution mode, or sampling mode");
    }
    return result;
}
struct Captured {
    DecodeGraphPeerBridge bridge;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    Captured(SpeculativeRank0Fixture& fixture,AcceptanceRoute route)
        : bridge(fixture.logits.execution.dev[0]->device,fixture.logits.execution.dev[1]->device) {
        fixture.logits.select(0);
        definition.capture(fixture.logits.execution.dev[0]->stream,[&] { fixture.pipeline(route); },
                           {&bridge,fixture.logits.execution.dev[1]->stream});
        executable.instantiate(definition);
        executable.upload(fixture.logits.execution.dev[0]->stream);
        fixture.logits.retire();
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


struct Measurement { double enqueue,wall,device; };
Measurement measure(SpeculativeRank0Fixture& fixture,AcceptanceRoute route,Captured* graph,
                    Timers& timers,ops::PeerMailbox& mailbox,bool mailbox_enabled) {
    // Same represented logits, frontier, RNG key, and stored occurrence counts for both routes.
    // Reset/poison is retired on BOTH GPUs before timing; no transfer or sampler setup is hidden
    // inside one route's interval. Target shards remain resident throughout the paired sweep.
    fixture.reset_counters(); fixture.controls(0,route); fixture.logits.retire();
    fixture.logits.reset_host_canaries();
    ops::PeerMailbox::reset_host_flags();
    if (graph) {
        fixture.logits.select(0);
        graph->bridge.gate_launch(fixture.logits.execution.dev[1]->stream,fixture.logits.execution.dev[0]->stream);
    }
    timers.start();
    const auto begin=std::chrono::steady_clock::now();
    if (graph) {
        fixture.logits.select(0); graph->executable.launch(fixture.logits.execution.dev[0]->stream);
    } else { fixture.pipeline(route); }
    const auto enqueued=std::chrono::steady_clock::now();
    const double device_us=timers.finish()*1000.0;
    const auto complete=std::chrono::steady_clock::now();
    const int used=graph && mailbox_enabled && route==AcceptanceRoute::RankZero ? 1 : 0;
    if (verify_acceptance_mailbox(mailbox,1,0,used)!=0) {
        throw std::runtime_error("measured route did not exercise its declared decision transport");
    }
    return {std::chrono::duration<double,std::micro>(enqueued-begin).count(),
            std::chrono::duration<double,std::micro>(complete-begin).count(),device_us};
}
void report(const std::string& mode,const std::string& sampling,AcceptanceRoute route,
            const std::vector<Measurement>& samples) {
    std::array<std::vector<double>,3> values;
    for (const auto& sample : samples) {
        values[0].push_back(sample.enqueue); values[1].push_back(sample.wall); values[2].push_back(sample.device);
    }
    for (auto& field : values) { std::sort(field.begin(),field.end()); }
    const auto median=samples.size()/2,p95=(samples.size()*95+99)/100-1;
    std::cout<<std::setprecision(9)<<"{\"type\":\"summary\",\"mode\":\""<<mode
             <<"\",\"sampling\":\""<<sampling<<"\",\"route\":\""<<acceptance_route_name(route)
             <<"\",\"samples\":"<<samples.size()<<",\"median_enqueue_us\":"<<values[0][median]
             <<",\"median_complete_wall_us\":"<<values[1][median]
             <<",\"median_joined_device_us\":"<<values[2][median]
             <<",\"p95_complete_wall_us\":"<<values[1][p95]
             <<",\"p95_joined_device_us\":"<<values[2][p95]<<"}\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const Options options=parse(argc,argv);
        if (cuda_unavailable()) { std::cout<<"SKIP: no CUDA device\n"; return 77; }
        int count=0; CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count<2) { std::cout<<"SKIP: two CUDA devices required\n"; return 77; }
        const std::vector<int> devices=options.reverse ? std::vector<int>{1,0} : std::vector<int>{0,1};
        AcceptanceMailboxEnvironment environment(options.mailbox);
        for (const std::string sampling : {std::string("greedy"),std::string("stochastic")}) {
            if (options.sampling!="both" && options.sampling!=sampling) { continue; }
            SpeculativeRank0Fixture fixture(devices,124160,124160,248077,3,1,1,
                                           sampling=="greedy" ? 0 : 2);
            fixture.prepare(29,true); fixture.logits.upload(0); fixture.logits.retire();
            ops::PeerMailbox mailbox(fixture.logits.execution,512,1);
            Timers timers(fixture.logits.execution);
            if (!fixture.logits.transfer.uses_host_staging({0,fixture.logits.storage[1]->part.bytes()}) ||
                !fixture.logits.eager_staged(ColumnGatherRoute::Packed)) {
                throw std::runtime_error("benchmark requires explicit pinned eager logit transport");
            }
            std::cout<<"{\"type\":\"configuration\",\"scope\":\"complete-two-rank-acceptance-pipeline\","
                       "\"physical_rows\":248320,\"token_domain\":248077,\"K\":3,\"B\":1,\"extent\":3,"
                       "\"input\":\"resident represented BF16 logits\",\"state_reset\":\"outside timing\","
                       "\"order\":\"rotating paired\",\"eager_gather\":\"pinned-staging\",\"devices\":["
                     <<devices[0]<<','<<devices[1]<<"],\"direct_p2p\":"<<(fixture.logits.direct_p2p ? "true" : "false")
                     <<",\"sampling\":\""<<sampling<<"\",\"warmup\":"<<options.warmup
                     <<",\"samples\":"<<options.samples<<",\"graph_decision_transport\":\""
                     <<(options.mailbox ? "mapped-mailbox" : "event-copy")<<"\"}\n";
            constexpr std::array<AcceptanceRoute,2> routes{AcceptanceRoute::Replicated,AcceptanceRoute::RankZero};
            for (const std::string mode : {std::string("eager"),std::string("graph")}) {
                if (options.mode!="both" && options.mode!=mode) { continue; }
                std::array<std::unique_ptr<Captured>,2> graphs;
                if (mode=="graph") {
                    for (int r=0; r<2; ++r) { graphs[r]=std::make_unique<Captured>(fixture,routes[r]); }
                }
                std::vector<int> reference;
                const auto validate=[&](int index) {
                    int failures=fixture.verify(routes[index],false);
                    if (routes[index]==AcceptanceRoute::RankZero) {
                        failures+=fixture.verify_root_transport(mode=="eager",0);
                    }
                    const auto actual=fixture.state_image(0,false);
                    if (reference.empty()) { reference=actual; }
                    else { failures+=verify_exact("complete baseline decision parity",actual,reference); }
                    if (failures) { throw std::runtime_error("acceptance pipeline benchmark qualification failed"); }
                };
                const auto qualify=[&] {
                    for (int r=0; r<2; ++r) {
                        (void)measure(fixture,routes[r],graphs[r].get(),timers,mailbox,options.mailbox);
                        validate(r);
                    }
                };
                qualify();
                for (int iteration=0; iteration<options.warmup; ++iteration) {
                    for (int item=0; item<2; ++item) {
                        const int r=(iteration+item)%2;
                        (void)measure(fixture,routes[r],graphs[r].get(),timers,mailbox,options.mailbox);
                    }
                }
                std::array<std::vector<Measurement>,2> samples;
                for (int iteration=0; iteration<options.samples; ++iteration) {
                    for (int item=0; item<2; ++item) {
                        const int r=(iteration+item)%2;
                        const auto value=measure(fixture,routes[r],graphs[r].get(),timers,mailbox,options.mailbox);
                        samples[r].push_back(value);
                        std::cout<<std::setprecision(9)<<"{\"type\":\"sample\",\"mode\":\""<<mode
                                 <<"\",\"sampling\":\""<<sampling<<"\",\"route\":\""<<acceptance_route_name(routes[r])
                                 <<"\",\"pair\":"<<iteration<<",\"order\":"<<item
                                 <<",\"enqueue_us\":"<<value.enqueue<<",\"complete_wall_us\":"<<value.wall
                                 <<",\"joined_device_us\":"<<value.device<<"}\n";
                        // Validate each route's resident timed-final state before any reset or
                        // poison can hide corruption. Qualification is outside its measured interval.
                        if (iteration==options.samples-1) { validate(r); }
                    }
                }
                qualify(); // Fresh poison and replay also reject inherited/no-op outputs.
                for (int r=0; r<2; ++r) { report(mode,sampling,routes[r],samples[r]); }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"rank-zero speculative benchmark: "<<error.what()<<'\n'; return 1;
    }
}
