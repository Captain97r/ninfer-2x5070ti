#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Production choices are unchanged. Alternative CTA grouping is benchmark-only.
    static constexpr int kWarpsPerCta = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    launch_nvfp4_swiglu_small_t<Geometry, ActiveTokens, kWarpsPerCta>(x, weight, out, stream);
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
constexpr auto kLaunchers =
    make_launchers<Geometry>(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kLaunchers<Nvfp4MlpGateUpGeometry>[index](x, weight, out, stream);
}

void nvfp4_linear_swiglu_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                              cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kLaunchers<Nvfp4MlpGateUpTp2ColumnGeometry>[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
