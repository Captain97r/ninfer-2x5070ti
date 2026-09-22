#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a4(cudaStream_t stream) {
    constexpr std::array attn_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gdn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gate_up_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array residual_invocations{
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, attn_invocations}, stream);
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, gdn_invocations}, stream);
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, true, gate_up_invocations}, stream);
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, residual_invocations}, stream);
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, residual_invocations}, stream);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // Match the engine's nonblocking streams. Default-stream tests cannot expose
        // descriptors freed on a different stream from their kernel consumer.
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            throw std::runtime_error("failed to create nonblocking test stream");
        }
        struct StreamOwner {
            cudaStream_t value;
            ~StreamOwner() { (void)cudaStreamDestroy(value); }
        } owner{stream};
        const int failures = run_nvfp4_a4(stream);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}
