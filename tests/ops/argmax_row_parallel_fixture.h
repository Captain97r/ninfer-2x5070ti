#pragma once

#include "core/decode_graph.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/speculative_round.h"
#include "ops/argmax_oracle.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::test {

class ArgmaxMailboxEnvironment {
public:
    explicit ArgmaxMailboxEnvironment(const char* value) {
        const char* previous = std::getenv("NINFER_TP2_MAILBOX");
        if (previous != nullptr) { previous_ = previous; had_previous_ = true; }
        set(value);
    }
    ~ArgmaxMailboxEnvironment() { set(had_previous_ ? previous_.c_str() : nullptr); }
private:
    static void set(const char* value) {
#ifdef _WIN32
        (void)_putenv_s("NINFER_TP2_MAILBOX", value == nullptr ? "" : value);
#else
        if (value != nullptr) { (void)setenv("NINFER_TP2_MAILBOX", value, 1); }
        else { (void)unsetenv("NINFER_TP2_MAILBOX"); }
#endif
    }
    bool had_previous_ = false;
    std::string previous_;
};

// Argmax-specific storage shared by exact tests and the old/new comparison benchmark.
// Every captured upload reads stable pinned storage, refreshed only after both ranks retire.
class ArgmaxRowParallelFixture {
public:
    struct RankStorage {
        GuardedDeviceBuffer logits;
        GuardedDeviceBuffer whole;
        GuardedDeviceBuffer scratch;
        WorkspaceArena arena;
        PinnedHostBuffer ingress;
        Tensor shard;
        Tensor gathered;

        RankStorage(int rows, int total_rows, int columns, int calls)
            : logits(static_cast<std::size_t>(rows) * columns * 2),
              whole(static_cast<std::size_t>(total_rows) * columns * 2),
              scratch(ops::argmax_row_parallel_workspace_capacity_bytes(rows, columns)),
              arena(DeviceSpan{scratch.data(), scratch.bytes()}),
              ingress(static_cast<std::size_t>(rows) * columns * calls * 2),
              shard(logits.data(), DType::BF16, {rows, columns}),
              gathered(whole.data(), DType::BF16, {total_rows, columns}) {}
    };

    ExecutionContext execution;
    ops::PeerEvents events;
    const std::array<int, 2> rows;
    const int total_rows;
    const int valid_rows;
    const int columns;
    const int calls;
    std::array<std::unique_ptr<RankStorage>, 2> storage;
    std::unique_ptr<GuardedDeviceBuffer> output;
    std::unique_ptr<GuardedDeviceBuffer> id_map;
    std::vector<std::int32_t> host_map;
    std::vector<std::vector<std::uint16_t>> logical;

    ArgmaxRowParallelFixture(std::vector<int> devices, int rows0, int rows1, int valid,
                              int count, int invocation_count)
        : execution(devices), events(execution), rows{rows0, rows1}, total_rows(rows0 + rows1),
          valid_rows(valid), columns(count), calls(invocation_count),
          host_map(static_cast<std::size_t>(total_rows)), logical(calls) {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            storage[rank] = std::make_unique<RankStorage>(rows[rank], total_rows, columns, calls);
        }
        select(0);
        output = std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(columns) * calls * 4);
        id_map = std::make_unique<GuardedDeviceBuffer>(host_map.size() * 4);
        for (int row = 0; row < total_rows; ++row) {
            host_map[row] = static_cast<std::int32_t>((static_cast<std::int64_t>(row) * 37 + 13) %
                                                     valid_rows);
        }
        id_map->copy_from_host(host_map.data(), host_map.size() * 4);
        // Guard initialization uses each device's legacy default stream.
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    ~ArgmaxRowParallelFixture() {
        // Normal callers retire before destruction. On an exception also keep storage alive
        // until already-issued work is done, without throwing from cleanup.
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            (void)cudaStreamSynchronize(execution.dev[rank]->stream);
        }
        (void)cudaSetDevice(execution.dev[0]->device);
        output.reset();
        id_map.reset();
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            storage[rank].reset();
        }
    }

    void select(int rank) const { CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device)); }
    void retire() const {
        select(1);
        execution.dev[1]->synchronize();
        select(0);
        execution.dev[0]->synchronize();
    }

    void prepare(int replay, bool adversarial) {
        for (int call = 0; call < calls; ++call) {
            auto& values = logical[call];
            values.resize(static_cast<std::size_t>(total_rows) * columns);
            for (int column = 0; column < columns; ++column) {
                const std::size_t base = static_cast<std::size_t>(column) * total_rows;
                for (int row = 0; row < total_rows; ++row) {
                    values[base + row] = f32_to_bf16(-32.0f +
                        static_cast<float>((row * 13ULL + column * 7ULL + call * 5ULL) % 1024) / 64);
                }
                auto put = [&](int row, std::uint16_t value) {
                    if (row >= 0 && row < valid_rows) { values[base + row] = value; }
                };
                const int winner = (replay * 7919 + call * 104729 + column * 37) % valid_rows;
                put(winner, f32_to_bf16(64.0f));
                if (adversarial) {
                    const int kind = (replay * calls + call + column) % 10;
                    if (kind == 1) {
                        put(std::min(rows[0] - 1, valid_rows - 1), f32_to_bf16(128.0f));
                        put(rows[0], f32_to_bf16(128.0f));
                    } else if (kind == 2) {
                        put(0, 0x7fc1u); put(valid_rows - 1, 0x7f80u);
                        put(0, 0x7fc1u);
                    } else if (kind == 3) {
                        put(rows[0], 0x7fc1u);
                        if (valid_rows > rows[0] + 1) { put(valid_rows - 1, f32_to_bf16(256.0f)); }
                    } else if (kind == 4 || kind == 9) {
                        for (int row = 0; row < valid_rows; ++row) {
                            values[base + row] = kind == 4 ? 0xff80u : 0x7fc1u;
                        }
                    } else if (kind == 5) {
                        put(std::min(1, valid_rows - 1), 0x7f80u); put(rows[0], 0x7f80u);
                    } else if (kind == 6) {
                        put(winner, f32_to_bf16(-1.0f));
                        put(0, 0x8000u); put(valid_rows - 1, 0x0000u);
                    } else if (kind == 7) {
                        put(std::min(1, valid_rows - 1), 0x7fc1u);
                    } else if (kind == 8) {
                        for (int row = rows[0]; row < valid_rows; ++row) {
                            values[base + row] = 0x7fc1u;
                        }
                        put(std::min(rows[0] - 1, valid_rows - 1), f32_to_bf16(128.0f));
                    }
                }
                for (int row = valid_rows; row < total_rows; ++row) { values[base + row] = 0x7f80u; }
            }
            for (int rank = 0; rank < 2; ++rank) {
                auto* pinned = static_cast<std::uint16_t*>(storage[rank]->ingress.data()) +
                               static_cast<std::size_t>(call) * rows[rank] * columns;
                const int offset = rank == 0 ? 0 : rows[0];
                for (int column = 0; column < columns; ++column) {
                    std::copy_n(values.data() + static_cast<std::size_t>(column) * total_rows + offset,
                                rows[rank], pinned + static_cast<std::size_t>(column) * rows[rank]);
                }
            }
        }
        select(0);
        CUDA_CHECK(cudaMemsetAsync(output->data(), 0xcd, output->bytes(), execution.dev[0]->stream));
        retire();
    }

    void upload(int call) {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            const auto* source = static_cast<const std::uint16_t*>(storage[rank]->ingress.data()) +
                                 static_cast<std::size_t>(call) * rows[rank] * columns;
            CUDA_CHECK(cudaMemcpyAsync(storage[rank]->logits.data(), source,
                                        storage[rank]->logits.bytes(), cudaMemcpyHostToDevice,
                                        execution.dev[rank]->stream));
        }
    }

    void issue(bool baseline, int call, bool upload_inputs, bool remap) {
        if (upload_inputs) { upload(call); }
        std::array<Tensor, 2> shards{storage[0]->shard, storage[1]->shard};
        Tensor out(static_cast<std::int32_t*>(output->data()) + call * columns,
                   DType::I32, {columns});
        if (baseline) {
            for (int column = 0; column < columns; ++column) {
                std::array<Tensor, 2> piece;
                std::array<Tensor, 2> full;
                for (int rank = 0; rank < 2; ++rank) {
                    piece[rank] = shards[rank].slice(1, column, 1).view({1, rows[rank]});
                    full[rank] = storage[rank]->gathered.slice(1, column, 1).view({1, total_rows});
                }
                ops::allgather_rows(full, piece, execution, events);
            }
            select(0);
            ops::argmax(storage[0]->gathered, out, valid_rows, execution.dev[0]->stream);
        } else {
            ops::argmax_row_parallel(shards, out, valid_rows,
                                     {&storage[0]->arena, &storage[1]->arena}, execution, events);
        }
        select(0);
        if (remap) {
            ops::proposal_remap_token_ids(out, static_cast<const std::int32_t*>(id_map->data()),
                                          total_rows, execution.dev[0]->stream);
        }
    }

    int verify(bool remap) {
        int failures = 0;
        std::vector<std::int32_t> expected;
        for (const auto& values : logical) {
            auto indices = argmax_oracle(values, total_rows, columns, valid_rows);
            for (const int index : indices) { expected.push_back(remap ? host_map[index] : index); }
        }
        select(0);
        failures += verify_exact("row-parallel argmax", from_device<std::int32_t>(
            output->data(), static_cast<std::size_t>(calls) * columns), expected);
        failures += output->verify_guards("argmax output");
        failures += id_map->verify_guards("argmax id map");
        failures += verify_exact("argmax preserves id map",
                                  from_device<std::int32_t>(id_map->data(), host_map.size()), host_map);
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            const auto* last = static_cast<const std::uint16_t*>(storage[rank]->ingress.data()) +
                               static_cast<std::size_t>(calls - 1) * rows[rank] * columns;
            const std::vector<std::uint16_t> expected_input(last, last + rows[rank] * columns);
            failures += verify_exact("argmax preserves shard",
                from_device<std::uint16_t>(storage[rank]->logits.data(), expected_input.size()),
                expected_input);
            failures += storage[rank]->logits.verify_guards("argmax shard");
            failures += storage[rank]->scratch.verify_guards("argmax workspace");
            failures += storage[rank]->whole.verify_guards("argmax gathered storage");
        }
        return failures;
    }
};

} // namespace ninfer::test
