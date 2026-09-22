#pragma once

#include "ops/allgather_columns_fixture.h"
#include "ops/argmax_oracle.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::test {

enum class AcceptanceRoute { Replicated, RankZero };
inline const char* acceptance_route_name(AcceptanceRoute route) {
    return route == AcceptanceRoute::Replicated ? "accept-on-both" : "accept-on-rank0";
}

class AcceptanceMailboxEnvironment {
public:
    explicit AcceptanceMailboxEnvironment(bool enabled) {
        const char* value = std::getenv("NINFER_TP2_MAILBOX");
        if (value) { previous_ = value; had_previous_ = true; }
        set(enabled ? "1" : "0");
    }
    ~AcceptanceMailboxEnvironment() { set(had_previous_ ? previous_.c_str() : nullptr); }
private:
    static void set(const char* value) {
#ifdef _WIN32
        (void)_putenv_s("NINFER_TP2_MAILBOX", value ? value : "");
#else
        if (value) { (void)setenv("NINFER_TP2_MAILBOX", value, 1); }
        else { (void)unsetenv("NINFER_TP2_MAILBOX"); }
#endif
    }
    bool had_previous_ = false;
    std::string previous_;
};

// mode 0: greedy; 1: mixed greedy and deterministic top-k=1 sampling;
// mode 2: mixed greedy and nondegenerate sampling. The latter uses exact old/new parity plus
// independent state/count invariants; probability correctness remains qualified by the existing
// FP64 public-distribution speculative_round suite, not by another GPU implementation.
class SpeculativeRank0Fixture {
public:
    struct Rank {
        GuardedDeviceBuffer state, target, drafts, extents, counts, configs;
        GuardedDeviceBuffer scratch, history, counts_history, target_history, source_history;
        WorkspaceArena arena;
        PinnedHostBuffer control, initial_counts, host_configs;
        Rank(int k, int batch, int domain, int calls, std::size_t capacity)
            : state(static_cast<std::size_t>(k+5)*batch*4),
              target(static_cast<std::size_t>(k+1)*batch*4),
              drafts(static_cast<std::size_t>(k)*batch*4), extents(batch*4),
              counts(static_cast<std::size_t>(domain)*batch*4),
              configs(static_cast<std::size_t>(batch)*sizeof(ops::SamplingConfig)),
              scratch(capacity), history(state.bytes()*calls),
              counts_history(counts.bytes()*calls), target_history(target.bytes()*calls),
              source_history(state.bytes()*calls), arena(DeviceSpan{scratch.data(), scratch.bytes()}),
              control((state.bytes()+drafts.bytes()+extents.bytes())*calls),
              initial_counts(counts.bytes()), host_configs(configs.bytes()) {}
    };

    AllgatherColumnsFixture logits;
    const int k, batch, domain, mode;
    std::array<std::unique_ptr<Rank>, 2> rank;
    std::vector<std::vector<int>> host_drafts, host_extents, initial_state;
    std::vector<int> initial_counts;
    std::vector<ops::SamplingConfig> config;

    SpeculativeRank0Fixture(std::vector<int> devices, int rows0, int rows1, int valid,
                            int drafts, int batch_size, int calls, int sampling_mode,
                            bool pinned = true)
        : logits(devices, rows0, rows1, (drafts+1)*batch_size, calls, pinned),
          k(drafts), batch(batch_size), domain(valid), mode(sampling_mode),
          host_drafts(calls), host_extents(calls), initial_state(calls),
          initial_counts(static_cast<std::size_t>(domain)*batch), config(batch) {
        const auto capacity = std::max(
            ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(domain,k,k,batch,batch),
            ops::speculative_replicate_decision_workspace_capacity_bytes(k,batch));
        for (int r=0; r<2; ++r) {
            logits.select(r);
            rank[r] = std::make_unique<Rank>(k,batch,domain,calls,capacity);
            for (auto* buffer : buffers(r)) { buffer->fill(0xcd); }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    ~SpeculativeRank0Fixture() {
        for (int r=0; r<2; ++r) {
            (void)cudaSetDevice(logits.execution.dev[r]->device);
            (void)cudaStreamSynchronize(logits.execution.dev[r]->stream);
        }
        for (int r=0; r<2; ++r) {
            (void)cudaSetDevice(logits.execution.dev[r]->device);
            rank[r].reset();
        }
    }
    std::vector<GuardedDeviceBuffer*> buffers(int r) const {
        auto& s=*rank[r];
        return {&s.state,&s.target,&s.drafts,&s.extents,&s.counts,&s.configs,&s.scratch,
                &s.history,&s.counts_history,&s.target_history,&s.source_history};
    }
    ops::SpeculativeDecisionView decision(int r) const {
        auto* p=static_cast<int*>(rank[r]->state.data());
        return {Tensor(p,DType::I32,{batch}),Tensor(p+batch,DType::I32,{batch}),
                Tensor(p+2*batch,DType::I32,{k+1,batch}),
                Tensor(p+(k+3)*batch,DType::I32,{batch}),
                Tensor(p+(k+4)*batch,DType::I32,{batch})};
    }
    int repeated_token(int epoch,int call,int b) const {
        // Alternate winning vocabulary shards, including the peer shard in the real-head case.
        const int half=domain/2;
        return ((call+b)%2)*half + 2 + (epoch*7+call*11+b*5)%(half-4);
    }
    void prepare(int epoch, bool full_extent = false) {
        logits.retire();
        std::fill(initial_counts.begin(),initial_counts.end(),0);
        for (int b=0; b<batch; ++b) {
            config[b]={};
            config[b].temperature=(mode!=0 && (batch==1 || b%2==1)) ? 0.8f : 0.0f;
            config[b].top_k=mode==2 ? 4 : 1;
            config[b].top_p=mode==2 ? 0.9f : 1.0f;
            config[b].min_p=mode==2 ? 0.04f : 0.0f;
            config[b].presence_penalty=1.5f;
            config[b].frequency_penalty=0.125f;
            config[b].seed=991u+static_cast<unsigned long long>(epoch)*31u+b;
            for (int call=0; call<logits.calls; ++call) {
                initial_counts[static_cast<std::size_t>(b)*domain+repeated_token(epoch,call,b)+1]=3;
            }
        }
        for (int call=0; call<logits.calls; ++call) {
            auto& d=host_drafts[call]; d.resize(k*batch);
            auto& e=host_extents[call]; e.resize(batch);
            auto& state=initial_state[call]; state.assign((k+5)*batch,-1907);
            for (int b=0; b<batch; ++b) {
                state[b]=137+epoch*13+call*37+b*19;
                e[b]=full_extent ? k : (call%3==0 ? 0 : (call%3==1 ? k : k-1));
                const int token=repeated_token(epoch,call,b);
                std::fill_n(d.begin()+b*k,k,token);
                for (int t=0; t<=k; ++t) {
                    const int winner=(call%3==2 && t>=1) ? token+2 : token;
                    for (int r=0; r<2; ++r) {
                        auto* out=const_cast<std::uint16_t*>(logits.host_part(r,call))+
                                  static_cast<std::size_t>(b*(k+1)+t)*logits.rows[r];
                        for (int v=0; v<logits.rows[r]; ++v) {
                            const int global=v+(r==0 ? 0 : logits.rows[0]);
                            out[v]=f32_to_bf16(-24.0f+static_cast<float>((global*13u+t*7u+b*5u+call)%97)/16.0f);
                            if (global==winner) { out[v]=f32_to_bf16(8.0f); }
                            if (global==token+1) { out[v]=f32_to_bf16(mode==0 ? 7.0f : 8.25f); }
                            // Exact relocation must preserve noncanonical NaNs and signed zero;
                            // none participates in argmax/sampling outside the logical domain.
                            if (global>=domain) {
                                constexpr std::uint16_t padding[]{0x7fc1,0xff81,0x8000,0x0000,0x7f80};
                                out[v]=padding[(global+t+call)%5];
                            }
                        }
                    }
                }
            }
            for (int r=0; r<2; ++r) {
                auto& s=*rank[r];
                auto* p=static_cast<int*>(s.control.data())+
                    call*(s.state.bytes()+s.drafts.bytes()+s.extents.bytes())/4;
                std::copy(state.begin(),state.end(),p); p+=state.size();
                std::copy(d.begin(),d.end(),p); p+=d.size();
                std::copy(e.begin(),e.end(),p);
            }
        }
        for (int r=0; r<2; ++r) {
            logits.select(r);
            auto& s=*rank[r];
            std::memcpy(s.initial_counts.data(),initial_counts.data(),s.counts.bytes());
            auto* cfg=static_cast<ops::SamplingConfig*>(s.host_configs.data());
            for (int b=0; b<batch; ++b) {
                cfg[b]=config[b];
                cfg[b].token_counts=b%4==3 ? nullptr : static_cast<int*>(s.counts.data())+b*domain;
            }
            CUDA_CHECK(cudaMemcpyAsync(s.configs.data(),s.host_configs.data(),s.configs.bytes(),
                cudaMemcpyHostToDevice,logits.execution.dev[r]->stream));
        }
        reset_counters();
        logits.retire();
    }
    void reset_counters() {
        for (int r=0; r<2; ++r) {
            logits.select(r);
            CUDA_CHECK(cudaMemcpyAsync(rank[r]->counts.data(),rank[r]->initial_counts.data(),
                rank[r]->counts.bytes(),cudaMemcpyHostToDevice,logits.execution.dev[r]->stream));
        }
    }
    void controls(int call, AcceptanceRoute route) {
        logits.poison_outputs();
        for (int r=0; r<2; ++r) {
            logits.select(r); auto& s=*rank[r]; auto stream=logits.execution.dev[r]->stream;
            const auto* p=static_cast<const std::uint8_t*>(s.control.data())+
                call*(s.state.bytes()+s.drafts.bytes()+s.extents.bytes());
            if (route==AcceptanceRoute::RankZero && r==1) {
                // The receiver's old frontier is deliberately unrelated. It must be overwritten,
                // not incremented, and may not be read to re-run acceptance or RNG.
                CUDA_CHECK(cudaMemsetAsync(s.state.data(),0xa7,s.state.bytes(),stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(s.state.data(),p,s.state.bytes(),cudaMemcpyHostToDevice,stream));
            }
            p+=s.state.bytes();
            CUDA_CHECK(cudaMemcpyAsync(s.drafts.data(),p,s.drafts.bytes(),cudaMemcpyHostToDevice,stream));
            p+=s.drafts.bytes();
            CUDA_CHECK(cudaMemcpyAsync(s.extents.data(),p,s.extents.bytes(),cudaMemcpyHostToDevice,stream));
            CUDA_CHECK(cudaMemsetAsync(s.target.data(),0xa7,s.target.bytes(),stream));
        }
    }
    void pipeline(AcceptanceRoute route,int source_snapshot=-1) {
        if (route==AcceptanceRoute::Replicated) {
            logits.gather(ColumnGatherRoute::Packed);
        } else {
            const std::array<Tensor,2> part{
                Tensor(logits.storage[0]->part.data(),DType::BF16,{logits.rows[0],logits.columns}),
                Tensor(logits.storage[1]->part.data(),DType::BF16,{logits.rows[1],logits.columns})};
            Tensor full(logits.storage[0]->output.data(),DType::BF16,{logits.total_rows,logits.columns});
            ops::gather_columns_to_rank0(full,part,logits.storage[0]->arena.get(),
                                       logits.execution,logits.transfer);
        }
        for (int r=0; r<(route==AcceptanceRoute::Replicated ? 2 : 1); ++r) {
            logits.select(r); auto& s=*rank[r]; const auto stream=logits.execution.dev[r]->stream;
            Tensor full(logits.storage[r]->output.data(),DType::BF16,{logits.total_rows,logits.columns});
            Tensor target(s.target.data(),DType::I32,{logits.columns});
            ops::argmax(full,target,domain,stream);
            auto out=decision(r);
            ops::speculative_accept_greedy_drafts(target.view({k+1,batch}),
                full.view({logits.total_rows,k+1,batch}),Tensor(s.drafts.data(),DType::I32,{k,batch}),
                Tensor(s.extents.data(),DType::I32,{batch}),out.frontiers,out.anchors,
                out.licensed_tokens,out.licensed_counts,out.accepted_drafts,domain,
                static_cast<const ops::SamplingConfig*>(s.configs.data()),s.arena,stream);
        }
        if (route==AcceptanceRoute::RankZero) {
            if (source_snapshot>=0) {
                logits.select(0); auto& s=*rank[0];
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(s.source_history.data())+
                    source_snapshot*s.state.bytes(),s.state.data(),s.state.bytes(),
                    cudaMemcpyDeviceToDevice,logits.execution.dev[0]->stream));
            }
            ops::speculative_replicate_decision({decision(0),decision(1)},
                static_cast<const ops::SamplingConfig*>(rank[1]->configs.data()),
                {&rank[0]->arena,&rank[1]->arena},logits.execution,logits.transfer);
        }
    }
    void issue(AcceptanceRoute route,int call) {
        logits.upload(call); controls(call,route); pipeline(route,call);
        for (int r=0; r<2; ++r) {
            logits.select(r); auto stream=logits.execution.dev[r]->stream;
            auto copy=[&](GuardedDeviceBuffer& dst,const GuardedDeviceBuffer& src) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(dst.data())+call*src.bytes(),
                    src.data(),src.bytes(),cudaMemcpyDeviceToDevice,stream));
            };
            auto& s=*rank[r]; auto& l=*logits.storage[r];
            copy(s.history,s.state); copy(s.counts_history,s.counts);
            copy(s.target_history,s.target); copy(l.input_history,l.part); copy(l.output_history,l.output);
        }
    }
    int verify_root_transport(bool staged,int call) const {
        int failures=0;
        for (int r=0; r<2; ++r) {
            const auto* bytes=static_cast<const std::uint8_t*>(logits.transfer.host_buffer(r));
            const std::size_t written=staged && r==1 ? logits.storage[1]->part.bytes() : 0;
            if (written) {
                const auto* values=reinterpret_cast<const std::uint16_t*>(bytes);
                const auto* expected=logits.host_part(1,call);
                failures+=verify_exact("one-way pinned logit publication",
                    std::vector<std::uint16_t>(values,values+written/2),
                    std::vector<std::uint16_t>(expected,expected+written/2));
            }
            for (std::size_t i=written; i<logits.transfer.host_capacity_bytes(); ++i) {
                if (bytes[i]!=0x6d) {
                    std::cerr<<"rank-zero gather touched unused pinned storage rank="<<r<<'\n';
                    ++failures; break;
                }
            }
        }
        return failures;
    }
    std::vector<int> state_image(int r,bool histories,int call=0) const {
        logits.select(r); const auto& s=*rank[r];
        const auto* p=static_cast<const std::uint8_t*>(histories ? s.history.data() : s.state.data())+
            (histories ? call*s.state.bytes() : 0);
        return from_device<int>(p,s.state.bytes()/4);
    }
    // Independent top-k=1 formula: decode represented BF16, apply public penalties with the
    // accepted draft prefix overlay, then scan the complete logical domain in increasing order.
    int deterministic_token(const std::vector<std::uint16_t>& image,const std::vector<int>& counts,
                            int call,int b,int column) const {
        int best=0; double maximum=-std::numeric_limits<double>::infinity();
        for (int v=0; v<domain; ++v) {
            double value=bf16_to_f32(image[(static_cast<std::size_t>(b)*(k+1)+column)*logits.total_rows+v]);
            if (config[b].temperature>0) {
                int count=b%4==3 ? 0 : counts[static_cast<std::size_t>(b)*domain+v];
                // The speculative prefix is present even without persistent occurrence storage.
                for (int i=0; i<column; ++i) { count+=host_drafts[call][b*k+i]==v; }
                value-=config[b].presence_penalty*(count>0)+config[b].frequency_penalty*count;
            }
            if (value>maximum) { maximum=value; best=v; }
        }
        return best;
    }
    int verify(AcceptanceRoute route,bool histories) const {
        int failures=0;
        std::vector<int> counts=initial_counts;
        const int count=histories ? logits.calls : 1;
        for (int call=0; call<count; ++call) {
            const auto image=logits.oracle(call);
            const auto targets=argmax_oracle(image,logits.total_rows,logits.columns,domain);
            const auto root=state_image(0,histories,call);
            const auto peer=state_image(1,histories,call);
            failures+=verify_exact("rank-zero decision replica",peer,root);
            for (int b=0; b<batch; ++b) {
                const int a=root[(k+4)*batch+b], n=root[(k+3)*batch+b];
                if (a<0 || a>host_extents[call][b] || n!=a+1 ||
                    root[b]!=initial_state[call][b]+n) {
                    std::cerr<<"invalid acceptance frontier/count\n"; ++failures; continue;
                }
                const int* licensed=root.data()+2*batch+b*(k+1);
                if (licensed[a]<0 || licensed[a]>=domain || root[batch+b]!=licensed[a]) { ++failures; }
                for (int i=0; i<=k; ++i) {
                    if ((i<a && licensed[i]!=host_drafts[call][b*k+i]) ||
                        (i>=n && licensed[i]!=0)) { ++failures; }
                }
                if (mode!=2 || config[b].temperature<=0) {
                    int expected_a=0;
                    while (expected_a<host_extents[call][b] &&
                           deterministic_token(image,counts,call,b,expected_a)==host_drafts[call][b*k+expected_a]) {
                        ++expected_a;
                    }
                    const int terminal=deterministic_token(image,counts,call,b,expected_a);
                    if (a!=expected_a || licensed[a]!=terminal) {
                        std::cerr<<"independent greedy/top-k1 decision mismatch row="<<b<<" call="<<call<<'\n';
                        ++failures;
                    }
                }
                if (config[b].temperature>0 && b%4!=3) {
                    for (int i=0; i<n; ++i) {
                        if (licensed[i]>=0 && licensed[i]<domain) {
                            ++counts[static_cast<std::size_t>(b)*domain+licensed[i]];
                        }
                    }
                }
            }
            for (int r=0; r<2; ++r) {
                logits.select(r); const auto& s=*rank[r]; const auto& l=*logits.storage[r];
                const auto offset=[&](const GuardedDeviceBuffer& live,const GuardedDeviceBuffer& history) {
                    return static_cast<const std::uint8_t*>(histories ? history.data() : live.data())+
                           (histories ? call*live.bytes() : 0);
                };
                failures+=verify_exact("licensed token occurrence counters",
                    from_device<int>(offset(s.counts,s.counts_history),counts.size()),counts);
                if (r==0 || route==AcceptanceRoute::Replicated) {
                    failures+=verify_exact("gathered logits raw BF16 bits",
                        from_device<std::uint16_t>(offset(l.output,l.output_history),image.size()),image);
                    failures+=verify_exact("target token independent argmax",
                        from_device<int>(offset(s.target,s.target_history),targets.size()),targets);
                }
                const auto* original=logits.host_part(r,call);
                failures+=verify_exact("input logits preserved",
                    from_device<std::uint16_t>(offset(l.part,l.input_history),l.part.bytes()/2),
                    std::vector<std::uint16_t>(original,original+l.part.bytes()/2));
            }
            if (histories && route==AcceptanceRoute::RankZero) {
                logits.select(0); const auto& s=*rank[0];
                failures+=verify_exact("source decision unchanged by replication",
                    from_device<int>(static_cast<const std::uint8_t*>(s.source_history.data())+
                                     call*s.state.bytes(),root.size()),root);
            }
        }
        for (int r=0; r<2; ++r) {
            logits.select(r); const auto& s=*rank[r]; const auto& l=*logits.storage[r];
            for (auto* buffer : buffers(r)) { failures+=buffer->verify_guards("acceptance guards"); }
            for (const auto* buffer : {&l.part,&l.output,&l.scratch,&l.input_history,&l.output_history}) {
                failures+=buffer->verify_guards("gather guards");
            }
            failures+=verify_exact("draft input unchanged",from_device<int>(s.drafts.data(),k*batch),host_drafts[count-1]);
            failures+=verify_exact("extent input unchanged",from_device<int>(s.extents.data(),batch),host_extents[count-1]);
            const auto* cfg=static_cast<const std::uint8_t*>(s.host_configs.data());
            failures+=verify_exact("sampling config unchanged",from_device<std::uint8_t>(s.configs.data(),s.configs.bytes()),
                                  std::vector<std::uint8_t>(cfg,cfg+s.configs.bytes()));
            if (s.arena.used()!=0 || s.arena.peak_used()>s.scratch.bytes() ||
                (l.arena && l.arena->used()!=0)) { ++failures; }
        }
        return failures;
    }
};

inline int verify_acceptance_mailbox(const ops::PeerMailbox& mailbox,int slots,int first,int used) {
    mailbox.validate_completed_round();
    int failures=0;
    for (int slot=0; slot<slots; ++slot) {
        const std::uint32_t expected=slot>=first && slot<first+used ? 1u : 0u;
        if (*mailbox.flag(0,slot)!=expected || *mailbox.flag(1,slot)!=0) {
            std::cerr<<"decision mailbox route/slot mismatch "<<slot<<'\n'; ++failures;
        }
    }
    return failures;
}

} // namespace ninfer::test
