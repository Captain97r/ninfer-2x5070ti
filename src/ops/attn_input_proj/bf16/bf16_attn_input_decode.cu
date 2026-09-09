#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"
#include <cuda_bf16.h>
namespace ninfer::ops::detail {
namespace {
template <int QueryRows, int KeyRows>
struct Bf16AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;
    __device__ __forceinline__ void store(std::int32_t parent_row, __nv_bfloat16 result) const {
        constexpr std::int32_t kQueryRows  = QueryRows;
        constexpr std::int32_t kKeyRows    = KeyRows;
        constexpr std::int32_t kGateRows   = QueryRows;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
        if (parent_row < kKeyBegin) {
            query[parent_row] = result;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result;
        } else {
            value[parent_row - kValueBegin] = result;
        }
    }
};
} // namespace
template <int Rows, int QueryRows, int KeyRows>
void launch_bf16_attn_input_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Bf16GemvGeometry<Rows, 5120>;
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;
    static_assert((QueryRows % Schedule::kRowsPerCta) == 0);
    static_assert((KeyRows % Schedule::kRowsPerCta) == 0);
    using Output = Bf16AttentionInputOutput<QueryRows, KeyRows>;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule, Output>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}
void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_bf16_attn_input_decode<14336, 6144, 1024>(x, weight, q, gate, k, v, stream);
}
void bf16_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_bf16_attn_input_decode<kBf16AttnInputShardRows, kBf16AttnInputShardQueryRows,
                                  kBf16AttnInputShardKeyRows>(x, weight, q, gate, k, v, stream);
}
} // namespace ninfer::ops::detail