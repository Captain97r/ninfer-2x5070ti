#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops::detail {
// The Attention epilogue has its own measured crossover. The wider candidate domain remains
// benchmark-callable so the production boundary is not conflated with template availability.
inline constexpr std::int32_t kBf16AttnInputSmallTMinTokens   = 2;
inline constexpr std::int32_t kBf16AttnInputSmallTMaxTokens   = 32;
inline constexpr std::int32_t kBf16AttnInputSmallTDispatchEnd = 22;
// The tp2 column shard: each device owns half the heads of every Q|K|Gate|V section (12
// query/gate heads, 2 key/value heads out of 24/4 total), in the same section order as the
// parent -- see include/ninfer/ops/attn_input_proj.h for the ShardPlan derivation.
inline constexpr std::int32_t kBf16AttnInputShardRows      = 7168;
inline constexpr std::int32_t kBf16AttnInputShardQueryRows = 3072;
inline constexpr std::int32_t kBf16AttnInputShardKeyRows   = 512;
void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream);
// --- TP2 column-shard siblings ([7168, 5120]) ----------------------------------------------------
// Same kernel templates as the tp1 functions above, instantiated at the shard geometry with the
// half-head section layout. Route selection (decode / small-T / mma by token count) is inherited
// unchanged from the tp1 parent.
void bf16_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v,
                                          cudaStream_t stream);
void bf16_attn_input_mma_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);
} // namespace ninfer::ops::detail