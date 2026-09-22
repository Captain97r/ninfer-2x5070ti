#pragma once

#include "core/decode_graph.h"
#include "ninfer/ops/allreduce.h"
#include "ops/op_tester.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace ninfer::test {

// Shared by the exact transport qualification and paired old/new timing. Every GPU/host allocation
// belongs to this fixture. Inputs for all calls stay pinned and immutable until both streams retire.
class PeerTransferFixture {
public:
    struct RankStorage {
        GuardedDeviceBuffer input;
        GuardedDeviceBuffer staging;
        GuardedDeviceBuffer gathered;
        GuardedDeviceBuffer sum_history;
        GuardedDeviceBuffer host_source_history;
        GuardedDeviceBuffer gather_history;
        GuardedDeviceBuffer skew;
        PinnedHostBuffer ingress;
        RankStorage(int elements, int gathered_elements, int calls)
            : input(static_cast<std::size_t>(elements) * 2),
              staging(static_cast<std::size_t>(elements) * 2),
              gathered(static_cast<std::size_t>(gathered_elements) * 2),
              sum_history(static_cast<std::size_t>(elements) * calls * 2),
              host_source_history(static_cast<std::size_t>(elements) * calls * 2),
              gather_history(static_cast<std::size_t>(gathered_elements) * calls * 2),
              skew(16u << 20),
              ingress(static_cast<std::size_t>(elements) * calls * 2) {}
    };

    static constexpr unsigned char host_canary = 0x6d;
    ExecutionContext execution;
    ops::PeerTransfer transfer;
    const int elements;
    const int calls;
    const std::array<int, 2> gather_rows;
    const int gathered_elements;
    std::array<std::unique_ptr<RankStorage>, 2> storage;

    PeerTransferFixture(const std::vector<int>& devices, int count, int invocation_count,
                         std::array<int, 2> rows, std::size_t host_capacity)
        : execution(devices), transfer(execution, host_capacity), elements(count),
          calls(invocation_count), gather_rows(rows), gathered_elements(rows[0] + rows[1]) {
        if (rows[0] <= 0 || rows[1] <= 0 || rows[0] > count || rows[1] > count) {
            throw std::invalid_argument("test gather halves must be positive input prefixes");
        }
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            storage[rank] = std::make_unique<RankStorage>(elements, gathered_elements, calls);
            storage[rank]->input.fill(0);
            storage[rank]->staging.fill(0xcd);
            storage[rank]->gathered.fill(0xcd);
            storage[rank]->sum_history.fill(0xcd);
            storage[rank]->host_source_history.fill(0xcd);
            storage[rank]->gather_history.fill(0xcd);
            storage[rank]->skew.fill(0);
            if (host_capacity != 0) {
                std::memset(transfer.host_buffer(rank), host_canary, host_capacity);
            }
            CUDA_CHECK(cudaDeviceSynchronize()); // retire default-stream guard initialization
        }
    }

    ~PeerTransferFixture() {
        // The transfer may have auxiliary source-copy work after a partially enqueued Op.
        // Retire its complete resource before freeing any input/staging allocation below.
        {
            ops::PeerTransfer retiring(std::move(transfer));
        }
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            (void)cudaStreamSynchronize(execution.dev[rank]->stream);
        }
        for (int rank = 0; rank < 2; ++rank) {
            (void)cudaSetDevice(execution.dev[rank]->device);
            storage[rank].reset();
        }
        (void)cudaSetDevice(execution.dev[0]->device);
    }

    void select(int rank) const { CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device)); }
    void retire() const {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            CUDA_CHECK(cudaStreamSynchronize(execution.dev[rank]->stream));
        }
    }

    void prepare(int epoch) {
        for (int rank = 0; rank < 2; ++rank) {
            auto* input = static_cast<std::uint16_t*>(storage[rank]->ingress.data());
            for (int call = 0; call < calls; ++call) {
                for (int i = 0; i < elements; ++i) {
                    // Bounded dyadic operands: the represented-input FP64 sum is also exactly
                    // representable in FP32. Only its final BF16 storage rounding is inexact.
                    const int value = static_cast<int>(
                        (static_cast<std::size_t>(i) * (rank == 0 ? 17 : 29) +
                         call * 47 + rank * 131 + epoch * 71) % 2048) - 1024;
                    input[static_cast<std::size_t>(call) * elements + i] =
                        f32_to_bf16(static_cast<float>(value) / 128.0f);
                }
            }
        }
    }

    void skew(int rank) {
        select(rank);
        for (int repeat = 0; repeat < 4; ++repeat) {
            CUDA_CHECK(cudaMemsetAsync(storage[rank]->skew.data(), repeat,
                                       storage[rank]->skew.bytes(), execution.dev[rank]->stream));
        }
    }

    void upload(int call, bool swap_rank_sources = false) {
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            const auto* source = static_cast<const std::uint16_t*>(
                storage[swap_rank_sources ? 1 - rank : rank]->ingress.data()) +
                static_cast<std::size_t>(call) * elements;
            CUDA_CHECK(cudaMemcpyAsync(storage[rank]->input.data(), source,
                                       static_cast<std::size_t>(elements) * 2,
                                       cudaMemcpyHostToDevice, execution.dev[rank]->stream));
        }
    }

    void sum(const ops::PeerTransfer& selected) {
        const std::array<Tensor, 2> input{
            Tensor(storage[0]->input.data(), DType::BF16, {elements}),
            Tensor(storage[1]->input.data(), DType::BF16, {elements})};
        const std::array<Tensor, 2> staging{
            Tensor(storage[0]->staging.data(), DType::BF16, {elements}),
            Tensor(storage[1]->staging.data(), DType::BF16, {elements})};
        ops::allreduce_sum(input, staging, execution, selected);
    }

    void issue(int call) {
        upload(call);
        sum(transfer);
        const std::size_t bytes = static_cast<std::size_t>(elements) * 2;
        const bool explicit_sum = transfer.uses_host_staging({bytes, bytes});
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            auto* output = static_cast<std::uint16_t*>(storage[rank]->sum_history.data()) +
                           static_cast<std::size_t>(call) * elements;
            CUDA_CHECK(cudaMemcpyAsync(output, storage[rank]->input.data(),
                                       static_cast<std::size_t>(elements) * 2,
                                       cudaMemcpyDeviceToDevice, execution.dev[rank]->stream));
            if (explicit_sum) {
                // Save the actual pinned source after sum retires on this stream and before
                // gather reuses it. An implicit sum cannot pass by letting only gather stage.
                auto* host_source = static_cast<std::uint16_t*>(
                    storage[rank]->host_source_history.data()) +
                    static_cast<std::size_t>(call) * elements;
                CUDA_CHECK(cudaMemcpyAsync(host_source, transfer.host_buffer(rank), bytes,
                                           cudaMemcpyHostToDevice, execution.dev[rank]->stream));
            }
        }
        // Gather distinct represented values on each GPU rather than the identical sum. Swap
        // the immutable ingress images between ranks: gather publication then differs from
        // the preceding sum publication AND the next call's upload, exposing wrong-rank reads,
        // skipped publication and cross-call source/pinned-buffer reuse without any host wait.
        upload(call, true);
        const std::array<Tensor, 2> destination{
            Tensor(storage[0]->gathered.data(), DType::BF16, {1, gathered_elements}),
            Tensor(storage[1]->gathered.data(), DType::BF16, {1, gathered_elements})};
        const std::array<Tensor, 2> part{
            Tensor(storage[0]->input.data(), DType::BF16, {1, gather_rows[0]}),
            Tensor(storage[1]->input.data(), DType::BF16, {1, gather_rows[1]})};
        ops::allgather_rows(destination, part, execution, transfer);
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            auto* output = static_cast<std::uint16_t*>(storage[rank]->gather_history.data()) +
                           static_cast<std::size_t>(call) * gathered_elements;
            CUDA_CHECK(cudaMemcpyAsync(output, storage[rank]->gathered.data(),
                                       static_cast<std::size_t>(gathered_elements) * 2,
                                       cudaMemcpyDeviceToDevice, execution.dev[rank]->stream));
        }
    }

    std::vector<std::uint16_t> sum_oracle(int call) const {
        const auto* a = static_cast<const std::uint16_t*>(storage[0]->ingress.data()) +
                        static_cast<std::size_t>(call) * elements;
        const auto* b = static_cast<const std::uint16_t*>(storage[1]->ingress.data()) +
                        static_cast<std::size_t>(call) * elements;
        std::vector<std::uint16_t> result(elements);
        for (int i = 0; i < elements; ++i) {
            const double exact_sum = static_cast<double>(bf16_to_f32(a[i])) +
                                     static_cast<double>(bf16_to_f32(b[i]));
            result[i] = f32_to_bf16(static_cast<float>(exact_sum));
        }
        return result;
    }

    int verify(bool explicit_route) {
        int failures = 0;
        std::vector<std::uint16_t> expected_sum;
        std::vector<std::uint16_t> expected_gather;
        for (int call = 0; call < calls; ++call) {
            const auto sum = sum_oracle(call);
            expected_sum.insert(expected_sum.end(), sum.begin(), sum.end());
            // Exact row-concatenation oracle from distinct represented inputs on each rank.
            for (int rank = 0; rank < 2; ++rank) {
                const auto* source = static_cast<const std::uint16_t*>(
                    storage[1 - rank]->ingress.data()) +
                    static_cast<std::size_t>(call) * elements;
                expected_gather.insert(expected_gather.end(), source,
                                       source + gather_rows[rank]);
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            select(rank);
            failures += verify_exact("peer transfer every sum",
                from_device<std::uint16_t>(storage[rank]->sum_history.data(), expected_sum.size()),
                expected_sum);
            failures += verify_exact("peer transfer every gather",
                from_device<std::uint16_t>(storage[rank]->gather_history.data(), expected_gather.size()),
                expected_gather);
            if (explicit_route) {
                const auto* input = static_cast<const std::uint16_t*>(storage[rank]->ingress.data());
                failures += verify_exact("peer transfer every pinned sum source",
                    from_device<std::uint16_t>(storage[rank]->host_source_history.data(),
                                                expected_sum.size()),
                    std::vector<std::uint16_t>(input, input + expected_sum.size()));
            }
            for (const auto* buffer : {&storage[rank]->input, &storage[rank]->staging,
                                      &storage[rank]->gathered, &storage[rank]->sum_history,
                                      &storage[rank]->host_source_history,
                                      &storage[rank]->gather_history, &storage[rank]->skew}) {
                failures += buffer->verify_guards("peer transfer device allocation");
            }
            const auto* host = static_cast<const unsigned char*>(transfer.host_buffer(rank));
            if (host == nullptr) { continue; }
            if (explicit_route) {
                // Confirms the route actually executed, rather than only checking its selector.
                const auto* values = reinterpret_cast<const std::uint16_t*>(host);
                const auto* source = static_cast<const std::uint16_t*>(
                    storage[1 - rank]->ingress.data()) +
                    static_cast<std::size_t>(calls - 1) * elements;
                failures += verify_exact("peer transfer pinned gather source",
                    std::vector<std::uint16_t>(values, values + gather_rows[rank]),
                    std::vector<std::uint16_t>(source, source + gather_rows[rank]));
            }
            const std::size_t canary_begin = explicit_route
                ? static_cast<std::size_t>(elements) * 2 : 0;
            for (std::size_t i = canary_begin; i < transfer.host_capacity_bytes(); ++i) {
                if (host[i] != host_canary) {
                    std::cerr << "peer transfer pinned buffer guard/fallback changed at " << i << '\n';
                    ++failures;
                    break;
                }
            }
        }
        return failures;
    }
};

} // namespace ninfer::test
