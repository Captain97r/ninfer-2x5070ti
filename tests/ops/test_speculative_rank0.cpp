#include "ops/speculative_rank0_fixture.h"

#include <string_view>

using namespace ninfer;
using namespace ninfer::test;
namespace {

void sequence(SpeculativeRank0Fixture& fixture,AcceptanceRoute route) {
    for (int call=0; call<fixture.logits.calls; ++call) {
        // Alternating stream delays expose a peer read racing the next source/scratch overwrite.
        fixture.logits.skew(call%2);
        fixture.issue(route,call);
    }
}
std::vector<int> decisions(const SpeculativeRank0Fixture& fixture) {
    std::vector<int> result;
    for (int call=0; call<fixture.logits.calls; ++call) {
        const auto state=fixture.state_image(0,true,call);
        result.insert(result.end(),state.begin(),state.end());
    }
    return result;
}
std::vector<int> baseline(SpeculativeRank0Fixture& fixture,int epoch,int& failures) {
    fixture.prepare(epoch);
    sequence(fixture,AcceptanceRoute::Replicated);
    fixture.logits.retire();
    failures+=fixture.verify(AcceptanceRoute::Replicated,true);
    return decisions(fixture);
}

int eager_case(const std::vector<int>& devices,int rows0,int rows1,int domain,int k,int batch,
               int mode,bool pinned) {
    SpeculativeRank0Fixture fixture(devices,rows0,rows1,domain,k,batch,3,mode,pinned);
    int failures=0;
    const bool expected_staged=pinned && static_cast<std::size_t>(rows1)*(k+1)*batch*2>=64u*1024u;
    if (fixture.logits.transfer.uses_host_staging({0,fixture.logits.storage[1]->part.bytes()})!=expected_staged) {
        throw std::runtime_error("rank-zero fixture selected wrong eager gather transport");
    }
    for (int epoch : {7,19}) {
        const auto expected=baseline(fixture,epoch,failures);
        fixture.prepare(epoch);
        fixture.logits.reset_host_canaries();
        sequence(fixture,AcceptanceRoute::RankZero);
        fixture.logits.retire();
        failures+=fixture.verify(AcceptanceRoute::RankZero,true);
        failures+=fixture.verify_root_transport(expected_staged,fixture.logits.calls-1);
        failures+=verify_exact("complete decision versus replicated acceptance",decisions(fixture),expected);
    }
    std::cout<<(failures ? "FAIL" : "OK")<<" eager rank0 acceptance V="<<domain<<" K="<<k
             <<" B="<<batch<<" sampling="<<mode<<" pinned="<<pinned<<'\n';
    return failures;
}

int graph_update_case(const std::vector<int>& devices,int rows0,int rows1,int domain,int k,int batch,
                      int mode,bool enabled) {
    AcceptanceMailboxEnvironment environment(enabled);
    SpeculativeRank0Fixture first(devices,rows0,rows1,domain,k,batch,3,mode);
    SpeculativeRank0Fixture second(devices,rows0,rows1,domain,k,batch,3,mode);
    constexpr int slots=6;
    ops::PeerMailbox mailbox(first.logits.execution,512,slots);
    DecodeGraphPeerBridge first_bridge(devices[0],devices[1]),second_bridge(devices[0],devices[1]);
    DecodeGraphDefinition first_definition,second_definition;
    DecodeGraphExecutable executable;
    const auto capture=[&](SpeculativeRank0Fixture& fixture,DecodeGraphPeerBridge& bridge,
                            DecodeGraphDefinition& definition) {
        fixture.prepare(23);
        fixture.logits.select(0);
        definition.capture(fixture.logits.execution.dev[0]->stream,
            [&] { sequence(fixture,AcceptanceRoute::RankZero); },
            {&bridge,fixture.logits.execution.dev[1]->stream});
    };
    capture(first,first_bridge,first_definition);
    capture(second,second_bridge,second_definition);
    first.logits.select(0); executable.instantiate(first_definition);
    int failures=0;
    const auto replay=[&](SpeculativeRank0Fixture& fixture,DecodeGraphPeerBridge& bridge,
                         int epoch,int first_slot) {
        const auto expected=baseline(fixture,epoch,failures);
        fixture.prepare(epoch);
        fixture.logits.reset_host_canaries();
        // Both owners' previous rounds have retired before the shared mailbox is reset.
        ops::PeerMailbox::reset_host_flags();
        fixture.logits.skew(1);
        fixture.logits.select(0);
        bridge.gate_launch(fixture.logits.execution.dev[1]->stream,fixture.logits.execution.dev[0]->stream);
        executable.launch(fixture.logits.execution.dev[0]->stream);
        fixture.logits.retire();
        failures+=verify_acceptance_mailbox(mailbox,slots,first_slot,enabled ? fixture.logits.calls : 0);
        failures+=fixture.verify(AcceptanceRoute::RankZero,true);
        failures+=fixture.verify_root_transport(false,fixture.logits.calls-1);
        failures+=verify_exact("updated graph complete decision versus baseline",decisions(fixture),expected);
    };
    replay(first,first_bridge,31,0);
    replay(first,first_bridge,37,0);
    first.logits.select(0); executable.update(second_definition);
    replay(second,second_bridge,43,3);
    second.logits.select(0); executable.update(first_definition);
    replay(first,first_bridge,47,0);
    std::cout<<(failures ? "FAIL" : "OK")<<" rank0 acceptance replay/whole-update V="<<domain
             <<" K="<<k<<" B="<<batch<<" sampling="<<mode<<" mailbox="<<enabled<<'\n';
    return failures;
}

// Acceptance has at least two verification columns, so exercise the gather's distinct zero-
// scratch T1 route separately with unequal shards and arbitrary represented BF16 bits.
int single_column(const std::vector<int>& devices) {
    AllgatherColumnsFixture first(devices,23,19,1,3,true);
    int failures=0;
    if (ops::gather_columns_to_rank0_workspace_capacity_bytes(19,1)!=0) { ++failures; }
    const auto enqueue=[](AllgatherColumnsFixture& fixture) {
        for (int call=0; call<fixture.calls; ++call) {
            fixture.upload(call); fixture.poison_outputs();
            const std::array<Tensor,2> parts{
                Tensor(fixture.storage[0]->part.data(),DType::BF16,{23,1}),
                Tensor(fixture.storage[1]->part.data(),DType::BF16,{19,1})};
            Tensor output(fixture.storage[0]->output.data(),DType::BF16,{42,1});
            ops::gather_columns_to_rank0(output,parts,nullptr,fixture.execution,fixture.transfer);
            fixture.select(0);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(fixture.storage[0]->output_history.data())+
                call*output.bytes(),output.data,output.bytes(),cudaMemcpyDeviceToDevice,fixture.execution.dev[0]->stream));
        }
    };
    const auto verify=[&](AllgatherColumnsFixture& fixture) {
        fixture.select(0);
        for (int call=0; call<fixture.calls; ++call) {
            const auto* p=static_cast<const std::uint8_t*>(fixture.storage[0]->output_history.data())+
                          call*fixture.storage[0]->output.bytes();
            failures+=verify_exact("asymmetric T1 root gather raw bits",from_device<std::uint16_t>(p,42),fixture.oracle(call));
        }
        for (int r=0; r<2; ++r) {
            fixture.select(r);
            const auto* original=fixture.host_part(r,fixture.calls-1);
            failures+=verify_exact("T1 source unchanged",from_device<std::uint16_t>(fixture.storage[r]->part.data(),fixture.rows[r]),
                                  std::vector<std::uint16_t>(original,original+fixture.rows[r]));
            for (const auto* buffer : {&fixture.storage[r]->part,&fixture.storage[r]->output,
                                      &fixture.storage[r]->output_history}) {
                failures+=buffer->verify_guards("T1 root gather guards");
            }
        }
    };
    first.prepare(13); enqueue(first); first.retire(); verify(first);
    AllgatherColumnsFixture second(devices,23,19,1,3,true);
    DecodeGraphPeerBridge first_bridge(devices[0],devices[1]),second_bridge(devices[0],devices[1]);
    DecodeGraphDefinition first_definition,second_definition;
    DecodeGraphExecutable executable;
    const auto capture=[&](AllgatherColumnsFixture& fixture,DecodeGraphPeerBridge& bridge,
                           DecodeGraphDefinition& definition) {
        fixture.prepare(17); fixture.select(0);
        definition.capture(fixture.execution.dev[0]->stream,[&] { enqueue(fixture); },
                           {&bridge,fixture.execution.dev[1]->stream});
    };
    capture(first,first_bridge,first_definition); capture(second,second_bridge,second_definition);
    first.select(0); executable.instantiate(first_definition);
    const auto replay=[&](AllgatherColumnsFixture& fixture,DecodeGraphPeerBridge& bridge,int epoch) {
        fixture.prepare(epoch); fixture.skew(1); fixture.select(0);
        bridge.gate_launch(fixture.execution.dev[1]->stream,fixture.execution.dev[0]->stream);
        executable.launch(fixture.execution.dev[0]->stream); fixture.retire(); verify(fixture);
    };
    replay(first,first_bridge,29); replay(first,first_bridge,31);
    first.select(0); executable.update(second_definition); replay(second,second_bridge,43);
    second.select(0); executable.update(first_definition); replay(first,first_bridge,47);
    std::cout<<(failures ? "FAIL" : "OK")<<" asymmetric T1 root gather eager/replay/whole-update\n";
    return failures;
}
} // namespace

int main(int argc,char** argv) {
    try {
        std::vector<int> devices{0,1};
        if (argc==2 && std::string_view(argv[1])=="--reverse-devices") { devices={1,0}; }
        else if (argc!=1) { throw std::invalid_argument("usage: ninfer_speculative_rank0_test [--reverse-devices]"); }
        if (cuda_unavailable()) { std::cout<<"SKIP: no CUDA device\n"; return 77; }
        int count=0; CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count<2) { std::cout<<"SKIP: two CUDA devices required\n"; return 77; }
        int failures=single_column(devices);
        for (bool pinned : {false,true}) {
            failures+=eager_case(devices,124160,124160,248077,3,1,0,pinned);
            failures+=eager_case(devices,124160,124160,248077,3,1,2,pinned);
        }
        for (int mode : {1,2}) { failures+=eager_case(devices,33,39,64,3,2,mode,true); }
        // Boundary decision sizes include null-counter stochastic rows and ordinary greedy rows.
        failures+=eager_case(devices,32,40,64,1,2,1,true);
        failures+=eager_case(devices,32,40,64,5,8,1,true);
        for (bool mailbox : {false,true}) {
            failures+=graph_update_case(devices,124160,124160,248077,3,1,0,mailbox);
            failures+=graph_update_case(devices,124160,124160,248077,3,1,2,mailbox);
            failures+=graph_update_case(devices,33,39,64,3,2,1,mailbox);
        }
        std::cout<<(failures ? "FAIL" : "OK")<<" rank-zero speculative decisions devices="
                 <<devices[0]<<','<<devices[1]<<'\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr<<"rank-zero speculative test: "<<error.what()<<'\n'; return 1;
    }
}
