// TP2 MTP prefix-state regression. Exact comparisons use matching 128-token prefill chunks:
// NVFP4 small-T decode and large-T prefill use different activation precision, so arbitrary
// decoded-prefix versus cold-prefill token equality is not a valid numerical contract.
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Logits = std::vector<std::uint16_t>;

ninfer::RequestOptions request(bool reuse, std::uint32_t outputs = 16) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = reuse;
    options.stop.include_model_defaults = false;
    return options;
}
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
float decode_bf16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
std::size_t agreement(const ninfer::GenerationResult& a, const ninfer::GenerationResult& b) {
    std::size_t i = 0;
    while (i < a.generated_token_ids.size() && i < b.generated_token_ids.size() &&
           a.generated_token_ids[i] == b.generated_token_ids[i]) { ++i; }
    return i;
}
void report_drift(const ninfer::GenerationResult& cached, const ninfer::GenerationResult& cold,
                  const Logits& cached_logits, const Logits& cold_logits, const char* label) {
    require(!cached_logits.empty() && cached_logits.size() == cold_logits.size(),
            "first-token logit capture is unavailable");
    double max_error = 0.0, error_sum = 0.0;
    for (std::size_t i = 0; i < cached_logits.size(); ++i) {
        const double error = std::abs(decode_bf16(cached_logits[i]) - decode_bf16(cold_logits[i]));
        max_error = std::max(max_error, error);
        error_sum += error * error;
    }
    const auto a = cached.generated_token_ids.front();
    const auto b = cold.generated_token_ids.front();
    std::cout << label << ": first-logit max_abs=" << max_error
              << " rms=" << std::sqrt(error_sum / cached_logits.size())
              << ", initial greedy agreement=" << agreement(cached, cold)
              << "/" << cached.generated_token_ids.size()
              << ", cached/cold first tokens=" << a << '/' << b
              << ", cached selection logits=" << decode_bf16(cached_logits[a]) << '/'
              << decode_bf16(cached_logits[b]) << ", cold selection logits="
              << decode_bf16(cold_logits[a]) << '/' << decode_bf16(cold_logits[b]) << '\n';
}
void compare_matched(const ninfer::GenerationResult& cached, const ninfer::GenerationResult& cold,
                     const Logits& cached_logits, const Logits& cold_logits, const char* label) {
    report_drift(cached, cold, cached_logits, cold_logits, label);
    require(cached.generated_token_ids.size() == 16 && cold.generated_token_ids.size() == 16,
            "prefix fixture did not generate 16 tokens");
    require(cached_logits == cold_logits,
            "matched-schedule prefix restoration changed first-token BF16 logits");
    require(cached.generated_token_ids == cold.generated_token_ids,
            "matched-schedule prefix restoration changed the greedy continuation");
    require(cached.speculative.rounds != 0 && cached.speculative.accepted_tokens != 0,
            "reused prefix did not exercise successful speculative decoding");
    require(cached.speculative.rounds == cold.speculative.rounds &&
                cached.speculative.accepted_tokens == cold.speculative.accepted_tokens &&
                cached.speculative.drafted_tokens == cold.speculative.drafted_tokens,
            "matched-schedule MTP proposal/acceptance state changed after reuse");
    std::cout << label << ": exact logits, 16/16 tokens and MTP counts agree; reused="
              << cached.reused_prompt_tokens << '\n';
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage out;
    out.role = role;
    out.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return out;
}
ninfer::PromptInput history(int responses) {
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User,
        "Use the lookup results to determine the deterministic checkpoint value."));
    for (int i = 0; i < responses; ++i) {
        const std::string key = i == 0 ? "alpha" : "beta";
        const std::string id = "call_" + key;
        auto assistant = message(ninfer::ChatRole::Assistant, "");
        assistant.reasoning_content = i == 0 ? "The first lookup should be alpha."
                                             : "The alpha result requests beta.";
        assistant.tool_calls.push_back(ninfer::ToolCall{
            .id = id, .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        input.messages.push_back(std::move(assistant));
        auto tool = message(ninfer::ChatRole::Tool,
                            i == 0 ? "{\"value\":17,\"next\":\"beta\"}" : "{\"value\":25}");
        tool.tool_call_id = id;
        input.messages.push_back(std::move(tool));
    }
    input.options.preserve_thinking = true;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
    return input;
}
void align_prompt(ninfer::Engine& engine, ninfer::PromptInput& input) {
    // Preserve existing history; pad only the latest user/tool content until the checkpoint is
    // a cold-prefill chunk boundary. This matches arithmetic without changing any runtime route.
    for (int i = 0; i < 256; ++i) {
        if (engine.count_tokens(input) % 128 == 0) { return; }
        input.messages.back().parts.front().text += " pad";
    }
    throw std::runtime_error("could not align the chat checkpoint to a 128-token boundary");
}
std::vector<ninfer::TokenId> synthetic(std::size_t count) {
    std::vector<ninfer::TokenId> tokens;
    for (std::size_t i = 0; i < count; ++i) {
        tokens.push_back(static_cast<ninfer::TokenId>(1000 + (i * 37) % 4096));
    }
    return tokens;
}

void exercise_decode_history(ninfer::Engine& engine) {
    // Keep the original 160-token + 16 decoded-token scenario visible. Cold replay uses W4A4
    // for historical tokens computed with A16 during decode, so report that numerical drift.
    // Replaying the SAME generation/append schedule must still reproduce tokens, logits and
    // acceptance exactly. Matched-chunk cases separately gate restoration against cold prefill.
    const auto prompt = synthetic(160);
    const auto seed = engine.generate(engine.prepare_tokens(prompt), request(false));
    require(seed.generated_token_ids.size() == 16 && seed.speculative.rounds != 0,
            "decoded-prefix source did not exercise MTP");
    auto continuation = prompt;
    continuation.insert(continuation.end(), seed.generated_token_ids.begin(), seed.generated_token_ids.end());
    for (int i = 0; i < 24; ++i) { continuation.push_back(2000 + i * 13); }
    const auto first = engine.generate(engine.prepare_tokens(continuation), request(true));
    const auto first_logits = engine.debug_last_round_logits_bf16();
    require(first.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier &&
                first.reused_prompt_tokens == 175,
            "decoded-prefix append did not resume the committed frontier");
    const auto cold = engine.generate(engine.prepare_tokens(continuation), request(false));
    const auto cold_logits = engine.debug_last_round_logits_bf16();
    report_drift(first, cold, first_logits, cold_logits, "decoded-prefix different-schedule diagnostic");
    const auto seed_repeat = engine.generate(engine.prepare_tokens(prompt), request(false));
    require(seed_repeat.generated_token_ids == seed.generated_token_ids,
            "decoded-prefix source is not reproducible");
    const auto repeat = engine.generate(engine.prepare_tokens(continuation), request(true));
    const auto repeat_logits = engine.debug_last_round_logits_bf16();
    compare_matched(first, repeat, first_logits, repeat_logits, "decoded-prefix same-schedule replay");
}

void exercise(const char* artifact, bool graphs) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = 2048;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk = 128;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = graphs;
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);
    engine.debug_enable_peer_egress_check(true);
    engine.debug_enable_logit_capture(true);
    std::cout << (graphs ? "graphs" : "eager") << ": matched 128-token prefill schedules\n";

    const auto prompt = synthetic(128);
    const auto source = engine.generate(engine.prepare_tokens(prompt), request(false, 1));
    require(source.generated_token_ids.size() == 1, "append source did not complete");
    auto continuation = prompt;
    continuation.push_back(source.generated_token_ids.front());
    for (int i = 0; i < 127; ++i) { continuation.push_back(2000 + i * 13); }
    const auto appended = engine.generate(engine.prepare_tokens(continuation), request(true));
    const auto appended_logits = engine.debug_last_round_logits_bf16();
    require(appended.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier &&
                appended.reused_prompt_tokens == 128,
            "TP2 MTP append did not reuse its matching chunk frontier");
    const auto append_cold = engine.generate(engine.prepare_tokens(continuation), request(false));
    const auto append_cold_logits = engine.debug_last_round_logits_bf16();
    compare_matched(appended, append_cold, appended_logits, append_cold_logits, "append versus cold");

    auto history0 = history(0);
    align_prompt(engine, history0);
    auto history1 = history(1);
    history1.messages[0] = history0.messages[0];
    align_prompt(engine, history1);
    auto history2 = history(2);
    history2.messages[0] = history0.messages[0];
    history2.messages[2] = history1.messages[2];
    align_prompt(engine, history2);
    const auto frontier0 = engine.count_tokens(history0);
    const auto frontier1 = engine.count_tokens(history1);
    std::cout << "aligned chat frontiers " << frontier0 << '/' << frontier1 << '/'
              << engine.count_tokens(history2) << '\n';

    // Sixteen generated tokens modify current GDN/KV state after each snapshot. Rewinding both
    // ranks then matches cold execution at the exact same chunk boundaries and arithmetic.
    const auto initial = engine.generate(engine.prepare(history0), request(false));
    require(initial.generated_token_ids.size() == 16, "checkpoint source did not complete");
    const auto checkpoint1 = engine.generate(engine.prepare(history1), request(true));
    const auto checkpoint1_logits = engine.debug_last_round_logits_bf16();
    require(checkpoint1.prefix_reuse_path == ninfer::PrefixReusePath::RestoreResponseCheckpoint &&
                checkpoint1.reused_prompt_tokens == frontier0,
            "first chat suffix did not restore the aligned response checkpoint");
    const auto checkpoint2 = engine.generate(engine.prepare(history2), request(true));
    const auto checkpoint2_logits = engine.debug_last_round_logits_bf16();
    require(checkpoint2.prefix_reuse_path == ninfer::PrefixReusePath::RestoreResponseCheckpoint &&
                checkpoint2.reused_prompt_tokens == frontier1,
            "rolling response checkpoint did not advance to the aligned frontier");
    const auto checkpoint2_cold = engine.generate(engine.prepare(history2), request(false));
    const auto checkpoint2_cold_logits = engine.debug_last_round_logits_bf16();
    compare_matched(checkpoint2, checkpoint2_cold, checkpoint2_logits, checkpoint2_cold_logits,
                    "rolling checkpoint versus cold");
    const auto checkpoint1_cold = engine.generate(engine.prepare(history1), request(false));
    const auto checkpoint1_cold_logits = engine.debug_last_round_logits_bf16();
    compare_matched(checkpoint1, checkpoint1_cold, checkpoint1_logits, checkpoint1_cold_logits,
                    "response checkpoint versus cold");

    exercise_decode_history(engine);
    const auto exact_source = engine.generate(engine.prepare_tokens(prompt), request(false, 1));
    const auto exact = engine.generate(engine.prepare_tokens(prompt), request(true, 1));
    require(exact.prefix_reuse_path == ninfer::PrefixReusePath::FullReset &&
                exact.reused_prompt_tokens == 0 &&
                exact.generated_token_ids == exact_source.generated_token_ids,
            "zero-suffix TP2 fallback changed behavior");
    const auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
    require(rounds != 0 && mismatches == 0, "TP2 MTP ranks disagree after prefix restoration");
    std::cout << (graphs ? "graphs" : "eager") << ": peer egress agrees in " << rounds << " rounds\n";
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    int count = 0;
    if (artifact == nullptr || *artifact == '\0' || cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS and two CUDA devices are required\n";
        return 77;
    }
    try {
        exercise(artifact, false);
        exercise(artifact, true);
    } catch (const std::exception& error) {
        std::cerr << "TP2 prefix regression: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}