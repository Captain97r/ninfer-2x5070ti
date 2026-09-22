#pragma once

#include "core/decode_graph.h"
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::test {

enum class ColumnGatherRoute { Baseline, Packed, Direct2D };

inline const char* column_gather_name(ColumnGatherRoute route) {
    switch (route) {
    case ColumnGatherRoute::Baseline: return "per-column";
    case ColumnGatherRoute::Packed: return "packed";
    case ColumnGatherRoute::Direct2D: return "direct2d-experiment";
    }
    throw std::invalid_argument("unknown gather route");
}

// Experimental direct-2D composition. It is deliberately outside the product Op: acceptance of
// peer 2D copies under no-P2P stream capture and whole-graph update must be measured separately.
// Inputs/resources are the validated fixture's; both destination copies use the owning stream.
inline void column_gather_direct2d(const std::array<Tensor, 2>& destination,
                                    const std::array<Tensor, 2>& part,
                                    const ExecutionContext& execution,
                                    const ops::PeerTransfer& transfer) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    struct Restore {
        int device;
        ~Restore() { (void)cudaSetDevice(device); }
    } restore{previous};
    const std::array<std::size_t, 2> bytes{part[0].bytes(), part[1].bytes()};
    const std::array<std::size_t, 2> width{
        static_cast<std::size_t>(part[0].ne[0]) * 2,
        static_cast<std::size_t>(part[1].ne[0]) * 2};
    const std::array<std::size_t, 2> offset{0, width[0]};
    const std::size_t pitch = width[0] + width[1];
    const auto height = static_cast<std::size_t>(part[0].ne[1]);
    const bool staged = transfer.uses_host_staging(bytes);
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *execution.dev[rank];
        CUDA_CHECK(cudaSetDevice(local.device));
        if (staged) {
            CUDA_CHECK(cudaMemcpyAsync(transfer.host_buffer(rank), part[rank].data, bytes[rank],
                                       cudaMemcpyDeviceToHost, local.stream));
        }
        CUDA_CHECK(cudaEventRecord(transfer.inputs_ready(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *execution.dev[rank];
        const int peer = 1 - rank;
        CUDA_CHECK(cudaSetDevice(local.device));
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.inputs_ready(peer), 0));
        auto* output = static_cast<std::uint8_t*>(destination[rank].data);
        CUDA_CHECK(cudaMemcpy2DAsync(output + offset[rank], pitch, part[rank].data, width[rank],
                                     width[rank], height, cudaMemcpyDeviceToDevice, local.stream));
        CUDA_CHECK(cudaMemcpy2DAsync(output + offset[peer], pitch,
                                     staged ? transfer.host_buffer(peer) : part[peer].data,
                                     width[peer], width[peer], height,
                                     staged ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,
                                     local.stream));
        CUDA_CHECK(cudaEventRecord(transfer.pull_done(rank), local.stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& local = *execution.dev[rank];
        CUDA_CHECK(cudaSetDevice(local.device));
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, transfer.pull_done(1-rank), 0));
    }
}

// Exact-storage fixture shared by qualification and the three-route comparison benchmark.
// Captured uploads read immutable pinned images until both devices are retired.
class AllgatherColumnsFixture {
public:
    struct Storage {
        GuardedDeviceBuffer part;
        GuardedDeviceBuffer output;
        GuardedDeviceBuffer scratch;
        std::unique_ptr<WorkspaceArena> arena;
        GuardedDeviceBuffer input_history;
        GuardedDeviceBuffer output_history;
        GuardedDeviceBuffer skew;
        PinnedHostBuffer ingress;

        Storage(int own_rows, int peer_rows, int columns, int calls)
            : part(static_cast<std::size_t>(own_rows) * columns * 2),
              output(static_cast<std::size_t>(own_rows + peer_rows) * columns * 2),
              scratch(ops::allgather_columns_workspace_capacity_bytes(peer_rows, columns)),
              arena(columns > 1 ? std::make_unique<WorkspaceArena>(
                  DeviceSpan{scratch.data(), scratch.bytes()}) : nullptr),
              input_history(part.bytes() * calls), output_history(output.bytes() * calls),
              skew(8u << 20), ingress(part.bytes() * calls) {}
    };

    ExecutionContext execution;
    ops::PeerTransfer transfer;
    const std::array<int, 2> rows;
    const int columns;
    const int calls;
    const int total_rows;
    const bool direct_p2p;
    std::array<std::unique_ptr<Storage>, 2> storage;

    AllgatherColumnsFixture(std::vector<int> devices, int rows0, int rows1, int width,
                             int invocation_count, bool provision_host)
        : execution(devices),
          transfer(execution, provision_host ?
              static_cast<std::size_t>(std::max(rows0, rows1)) * width * 2 + 256 : 0),
          rows{rows0, rows1}, columns(width), calls(invocation_count), total_rows(rows0 + rows1),
          direct_p2p(ops::enable_peer_access(execution)) {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            storage[rank] = std::make_unique<Storage>(rows[rank], rows[1-rank], columns, calls);
            storage[rank]->part.fill(0xcd);
            storage[rank]->output.fill(0xcd);
            storage[rank]->scratch.fill(0xcd);
            storage[rank]->input_history.fill(0xcd);
            storage[rank]->output_history.fill(0xcd);
            CUDA_CHECK(cudaDeviceSynchronize()); // retire legacy-stream guard initialization
        }
        reset_host_canaries();
    }

    ~AllgatherColumnsFixture() {
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            (void)cudaStreamSynchronize(execution.dev[rank]->stream);
        }
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            storage[rank].reset();
        }
    }

    void select(int rank) const { CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device)); }
    void retire() const {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaStreamSynchronize(execution.dev[rank]->stream));
        }
    }
    void reset_host_canaries() {
        for (int rank = 0; rank < 2; ++rank) {
            if (transfer.host_capacity_bytes() != 0) {
                std::memset(transfer.host_buffer(rank), 0x6d, transfer.host_capacity_bytes());
            }
        }
    }

    void prepare(int epoch) {
        constexpr std::array<std::uint16_t, 12> special{
            0x0000, 0x8000, 0x7f80, 0xff80, 0x7f81, 0x7fc1,
            0xff81, 0xffc7, 0x0001, 0x8001, 0x7f7f, 0xff7f};
        for (int rank = 0; rank < 2; ++rank) {
            auto* values = static_cast<std::uint16_t*>(storage[rank]->ingress.data());
            for (int call = 0; call < calls; ++call) {
                for (int column = 0; column < columns; ++column) {
                    for (int row = 0; row < rows[rank]; ++row) {
                        const std::size_t at = (static_cast<std::size_t>(call) * columns + column) *
                                               rows[rank] + row;
                        std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                            static_cast<std::uint32_t>(column) * 1013904223u +
                            static_cast<std::uint32_t>(call) * 2246822519u +
                            static_cast<std::uint32_t>(epoch) * 3266489917u + rank * 668265263u;
                        mixed ^= mixed >> 13;
                        values[at] = static_cast<std::uint16_t>(mixed);
                        // Both shard edges include arbitrary NaN payloads and signed zeros.
                        if (row < 12 || row >= rows[rank] - 12) {
                            values[at] = special[(static_cast<std::size_t>(row) + rank * 5 +
                                                  column + call + epoch) % special.size()];
                        }
                    }
                }
            }
        }
    }

    const std::uint16_t* host_part(int rank, int call) const {
        return static_cast<const std::uint16_t*>(storage[rank]->ingress.data()) +
               static_cast<std::size_t>(call) * rows[rank] * columns;
    }
    void upload(int call) {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaMemcpyAsync(storage[rank]->part.data(), host_part(rank, call),
                                       storage[rank]->part.bytes(), cudaMemcpyHostToDevice,
                                       execution.dev[rank]->stream));
        }
    }
    void poison_outputs() {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaMemsetAsync(storage[rank]->output.data(), 0xa7,
                                       storage[rank]->output.bytes(), execution.dev[rank]->stream));
        }
    }
    void skew(int rank) {
        select(rank);
        for (int repeat = 0; repeat < 4; ++repeat) {
            CUDA_CHECK(cudaMemsetAsync(storage[rank]->skew.data(), repeat,
                                       storage[rank]->skew.bytes(), execution.dev[rank]->stream));
        }
    }

    void gather(ColumnGatherRoute route) {
        const std::array<Tensor, 2> parts{
            Tensor(storage[0]->part.data(), DType::BF16, {rows[0], columns}),
            Tensor(storage[1]->part.data(), DType::BF16, {rows[1], columns})};
        const std::array<Tensor, 2> destination{
            Tensor(storage[0]->output.data(), DType::BF16, {total_rows, columns}),
            Tensor(storage[1]->output.data(), DType::BF16, {total_rows, columns})};
        if (route == ColumnGatherRoute::Packed) {
            const std::array<WorkspaceArena*, 2> workspace{
                storage[0]->arena.get(), storage[1]->arena.get()};
            ops::allgather_columns(destination, parts, workspace, execution, transfer);
        } else if (route == ColumnGatherRoute::Direct2D) {
            column_gather_direct2d(destination, parts, execution, transfer);
        } else {
            for (int column = 0; column < columns; ++column) {
                const std::array<Tensor, 2> piece{
                    parts[0].slice(1, column, 1).view({1, rows[0]}),
                    parts[1].slice(1, column, 1).view({1, rows[1]})};
                const std::array<Tensor, 2> whole{
                    destination[0].slice(1, column, 1).view({1, total_rows}),
                    destination[1].slice(1, column, 1).view({1, total_rows})};
                ops::allgather_rows(whole, piece, execution, transfer);
            }
        }
    }

    void issue(ColumnGatherRoute route, int call) {
        upload(call);
        poison_outputs();
        gather(route);
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            auto& item = *storage[rank];
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::uint8_t*>(item.input_history.data()) + call * item.part.bytes(),
                item.part.data(), item.part.bytes(), cudaMemcpyDeviceToDevice,
                execution.dev[rank]->stream));
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::uint8_t*>(item.output_history.data()) + call * item.output.bytes(),
                item.output.data(), item.output.bytes(), cudaMemcpyDeviceToDevice,
                execution.dev[rank]->stream));
        }
    }

    std::vector<std::uint16_t> oracle(int call) const {
        std::vector<std::uint16_t> result;
        result.reserve(static_cast<std::size_t>(total_rows) * columns);
        for (int column = 0; column < columns; ++column) {
            for (int rank = 0; rank < 2; ++rank) {
                const auto* source = host_part(rank, call) + static_cast<std::size_t>(column) * rows[rank];
                result.insert(result.end(), source, source + rows[rank]);
            }
        }
        return result;
    }

    int verify(bool histories, int resident_call = 0) const {
        std::vector<std::uint16_t> expected;
        if (histories) {
            for (int call = 0; call < calls; ++call) {
                const auto image = oracle(call);
                expected.insert(expected.end(), image.begin(), image.end());
            }
        } else {
            expected = oracle(resident_call);
        }
        int failures = 0;
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            const auto& item = *storage[rank];
            const auto* output = histories ? item.output_history.data() : item.output.data();
            failures += verify_exact("column gather every output bit",
                from_device<std::uint16_t>(output, expected.size()), expected);
            const std::size_t count = static_cast<std::size_t>(rows[rank]) * columns *
                                      (histories ? calls : 1);
            const auto* input = histories ? item.input_history.data() : item.part.data();
            const auto* original = host_part(rank, histories ? 0 : resident_call);
            failures += verify_exact("column gather input unchanged",
                from_device<std::uint16_t>(input, count),
                std::vector<std::uint16_t>(original, original + count));
            for (const auto* buffer : {&item.part, &item.output, &item.scratch, &item.input_history,
                                      &item.output_history, &item.skew}) {
                failures += buffer->verify_guards("column gather guarded allocation");
            }
        }
        return failures;
    }

    bool eager_staged(ColumnGatherRoute route) const {
        const int count = route == ColumnGatherRoute::Baseline ? 1 : columns;
        return transfer.uses_host_staging({static_cast<std::size_t>(rows[0]) * count * 2,
                                           static_cast<std::size_t>(rows[1]) * count * 2});
    }
    int verify_transport(ColumnGatherRoute route, bool staged, int last_call) const {
        if (transfer.host_capacity_bytes() == 0) { return 0; }
        int failures = 0;
        for (int rank = 0; rank < 2; ++rank) {
            const auto* bytes = static_cast<const std::uint8_t*>(transfer.host_buffer(rank));
            std::size_t written = 0;
            if (staged) {
                const bool columnwise = route == ColumnGatherRoute::Baseline;
                const auto* expected = host_part(rank, last_call) +
                    (columnwise ? static_cast<std::size_t>(columns - 1) * rows[rank] : 0);
                const std::size_t count = static_cast<std::size_t>(rows[rank]) *
                                          (columnwise ? 1 : columns);
                const auto* values = reinterpret_cast<const std::uint16_t*>(bytes);
                failures += verify_exact("column gather actual pinned publication",
                    std::vector<std::uint16_t>(values, values + count),
                    std::vector<std::uint16_t>(expected, expected + count));
                written = count * 2;
            }
            for (std::size_t i = written; i < transfer.host_capacity_bytes(); ++i) {
                if (bytes[i] != 0x6d) {
                    std::cerr << "column gather pinned guard/fallback changed at " << i << '\n';
                    ++failures;
                    break;
                }
            }
        }
        return failures;
    }
};

} // namespace ninfer::test
