#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/peer_mailbox.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

// Publish the same timeout report as peer_exchange_sum_kernel without waiting for a
// missing peer or risking the Windows watchdog. The write still occurs asynchronously
// on the GPU, inside the same cross-device graph used by decode.
__global__ void publish_timeout(volatile std::uint32_t* fault, bool timeout) {
    if (timeout) {
        *fault = 1;
        __threadfence_system();
    }
}

bool detects_timeout(const ninfer::ops::PeerMailbox& mailbox) {
    try {
        mailbox.validate_completed_round();
        return false;
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) !=
            "TP2 mailbox exchange timed out waiting for the peer device") {
            throw;
        }
        return true;
    }
}

void check_round(ninfer::ExecutionContext& execution, ninfer::ops::PeerMailbox& mailbox,
                 int fault_rank) {
    ninfer::ops::PeerMailbox::reset_host_flags();
    ninfer::DecodeGraphPeerBridge bridge(execution.dev[0]->device, execution.dev[1]->device);
    ninfer::DecodeGraphDefinition definition;
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    definition.capture(
        execution.dev[0]->stream,
        [&] {
            for (int rank = 0; rank < 2; ++rank) {
                CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                publish_timeout<<<1, 1, 0, execution.dev[rank]->stream>>>(mailbox.hang_word(),
                                                                        rank == fault_rank);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        },
        {&bridge, execution.dev[1]->stream});
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definition);
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    executable.launch(execution.dev[0]->stream);
    execution.dev[1]->synchronize();
    execution.dev[0]->synchronize();

    // Exactly ONE replay: a final decode round must fail at retirement even if the next
    // launch-side guard will never run. A healthy round must remain accepted.
    const bool expected = fault_rank >= 0;
    if (detects_timeout(mailbox) != expected) {
        throw std::runtime_error("mailbox fault was not validated on the completed round");
    }
    // Validation must not erase a failed reduction and allow a later consumer to accept it.
    if (detects_timeout(mailbox) != expected) {
        throw std::runtime_error("mailbox validation erased the completed round's fault");
    }
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count < 2)) {
        std::cout << "SKIP: mailbox completion test needs two CUDA devices\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        ninfer::ExecutionContext execution({0, 1});
        ninfer::ops::PeerMailbox mailbox(execution, 256, 1);
        check_round(execution, mailbox, -1);
        check_round(execution, mailbox, 0);
        check_round(execution, mailbox, 1);
        check_round(execution, mailbox, -1);
        std::cout << "ok: clean and faulted final graph rounds validated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mailbox completion test failed: " << error.what() << '\n';
        return 1;
    }
}
