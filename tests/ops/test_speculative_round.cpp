#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void initialize(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

DeviceBuffer device_config(const ops::SamplingConfig& config) {
    return to_device(std::vector<ops::SamplingConfig>{config});
}

struct VerifyInputsExpected {
    std::vector<std::int32_t> verify_ids;
    std::vector<std::int32_t> positions;
};

VerifyInputsExpected verify_inputs_oracle(std::int32_t token,
                                          const std::vector<std::int32_t>& drafts,
                                          std::int32_t length) {
    VerifyInputsExpected expected{
        .verify_ids = std::vector<std::int32_t>(drafts.size() + 1),
        .positions  = std::vector<std::int32_t>(drafts.size() + 1),
    };
    expected.verify_ids[0] = token;
    for (std::size_t i = 0; i < drafts.size(); ++i) expected.verify_ids[i + 1] = drafts[i];
    for (std::size_t i = 0; i < expected.positions.size(); ++i) {
        expected.positions[i] = length + static_cast<std::int32_t>(i);
    }
    return expected;
}

int prepare_verify_case(int k) {
    const std::int32_t token_value  = 70000 + k;
    const std::int32_t length_value = 1000 - k;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) drafts[static_cast<std::size_t>(i)] = 37 + 7919 * i;
    const auto expected = verify_inputs_oracle(token_value, drafts, length_value);

    DeviceBuffer d_token  = to_device<std::int32_t>({token_value});
    DeviceBuffer d_drafts = to_device(drafts);
    DeviceBuffer d_length = to_device<std::int32_t>({length_value});
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    GuardedDeviceBuffer d_verify(expected.verify_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(expected.positions.size() * sizeof(std::int32_t));
    d_verify.fill(0xcd);
    d_positions.fill(0xef);

    Tensor token(d_token.p, DType::I32, {1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor verify(d_verify.data(), DType::I32, {k + 1});
    Tensor positions(d_positions.data(), DType::I32, {k + 1});
    ops::speculative_prepare_verify_inputs(token, draft_tensor, length, extent, verify, positions,
                                           nullptr);
    cuda_synchronize();

    const std::string label = "speculative prepare K=" + std::to_string(k);
    int failures =
        verify_exact((label + " verify ids").c_str(),
                     read<std::int32_t>(d_verify, expected.verify_ids.size()), expected.verify_ids);
    failures += verify_exact((label + " positions").c_str(),
                             read<std::int32_t>(d_positions, expected.positions.size()),
                             expected.positions);
    failures += verify_exact((label + " token unchanged").c_str(),
                             from_device<std::int32_t>(d_token, 1), {token_value});
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(d_length, 1), {length_value});
    failures += d_verify.verify_guards((label + " verify guards").c_str());
    failures += d_positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
    std::int32_t accepted;
    std::int32_t length;
    std::int32_t token;
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
        .accepted    = accepted,
        .length      = initial_length + accepted + 1,
        .token       = terminal_token,
    };
    for (int i = 0; i < accepted; ++i) {
        expected.sampled[static_cast<std::size_t>(i)] = drafts[static_cast<std::size_t>(i)];
    }
    expected.sampled[static_cast<std::size_t>(accepted)] = terminal_token;
    return expected;
}

int execute_accept_case(const std::string& label, const std::vector<std::int32_t>& target_tokens,
                        const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                        const std::vector<std::int32_t>& drafts, std::int32_t initial_length,
                        int token_domain, ops::SamplingConfig config,
                        const std::vector<std::int32_t>& initial_token_counts,
                        const AcceptExpected& expected) {
    const int k              = static_cast<int>(drafts.size());
    DeviceBuffer d_targets   = to_device(target_tokens);
    DeviceBuffer d_logits    = to_device(logits_bits);
    DeviceBuffer d_drafts    = to_device(drafts);
    DeviceBuffer d_counts    = to_device(initial_token_counts);
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_token(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_token, std::vector<std::int32_t>{-1234567});
    d_sampled.fill(0x9d);
    initialize(d_num, std::vector<std::int32_t>{-11});
    initialize(d_accepted, std::vector<std::int32_t>{-13});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor token(d_token.data(), DType::I32, {1});
    Tensor sampled(d_sampled.data(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_greedy_drafts(
        targets, logits, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " sampled").c_str(), read<std::int32_t>(d_sampled, k + 1),
                                expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(), read<std::int32_t>(d_num, 1),
                             {expected.num_sampled});
    failures += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                             {expected.accepted});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {expected.length});
    failures +=
        verify_exact((label + " token").c_str(), read<std::int32_t>(d_token, 1), {expected.token});

    failures +=
        verify_exact((label + " target tokens unchanged").c_str(),
                     from_device<std::int32_t>(d_targets, target_tokens.size()), target_tokens);
    failures += verify_exact((label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " config unchanged").c_str(),
                             from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig)),
                             config_before);

    auto expected_counts = initial_token_counts;
    if (config.temperature > 0.0f) {
        for (int i = 0; i < expected.num_sampled; ++i) {
            ++expected_counts[static_cast<std::size_t>(
                expected.sampled[static_cast<std::size_t>(i)])];
        }
    }
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_token.verify_guards((label + " token guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    failures += d_accepted.verify_guards((label + " accepted guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_accept_case(int k, int accepted_count, int token_domain = 64) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(
        drafts, accepted_count, targets[static_cast<std::size_t>(accepted_count)], initial_length);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) token_counts[static_cast<std::size_t>(i)] = i % 5;
    return execute_accept_case("speculative greedy K=" + std::to_string(k) +
                                   " A=" + std::to_string(accepted_count),
                               targets, logits, token_domain, drafts, initial_length, token_domain,
                               ops::SamplingConfig{}, token_counts, expected);
}

int deterministic_sampling_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) targets[static_cast<std::size_t>(i)] = 101 + i;
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) logits_bits[i] = f32_to_bf16(logits[i]);

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    return execute_accept_case("speculative sampling deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int batched_sampling_workspace_stride_case() {
    constexpr int physical_rows = 257;
    constexpr int token_domain  = 257;
    constexpr int k             = 3;
    constexpr int batch         = 2;
    constexpr int columns       = k + 1;

    const std::vector<std::int32_t> drafts{10, 11, 12, 30, 31, 32};
    const std::vector<std::int32_t> winners{10, 20, 21, 22, 30, 31, 32, 33};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < columns; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + col) * physical_rows;
            logits[base + static_cast<std::size_t>(winners[row * columns + col])] =
                f32_to_bf16(20.0f);
        }
    }

    DeviceBuffer d_targets = to_device(winners);
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(workspace_bytes);
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 3});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {102, 204});
    failures += verify_exact("speculative sampling B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {20, 33});
    return failures;
}

// This oracle starts from the represented BF16 logits and evaluates the public
// sampling formula in FP64. It does not call the production sampler or copy its
// RNG, reduction tree, workspace layout, or rejection-sampling implementation.
struct OracleCandidate {
    int token;
    double probability;
};

std::vector<OracleCandidate> verification_distribution(
    std::span<const std::uint16_t> logits, const ops::SamplingConfig& config,
    std::span<const std::int32_t> initial_counts, std::span<const std::int32_t> prior_drafts) {
    struct ScoredToken {
        int token;
        double score;
    };
    std::vector<ScoredToken> sorted;
    sorted.reserve(logits.size());
    for (std::size_t token = 0; token < logits.size(); ++token) {
        int count = initial_counts.empty() ? 0 : initial_counts[token];
        count += static_cast<int>(std::count(prior_drafts.begin(), prior_drafts.end(),
                                            static_cast<std::int32_t>(token)));
        double score = bf16_to_f32(logits[token]);
        if (count > 0) { score -= static_cast<double>(config.presence_penalty); }
        score -= static_cast<double>(config.frequency_penalty) * count;
        sorted.push_back({static_cast<int>(token), score});
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.score == b.score ? a.token < b.token : a.score > b.score;
    });
    const int cap = std::min(static_cast<int>(sorted.size()),
                            config.top_k > 0 && config.top_k < 20 ? config.top_k : 20);
    sorted.resize(static_cast<std::size_t>(cap));
    std::vector<double> weights(static_cast<std::size_t>(cap));
    double total = 0.0;
    for (int i = 0; i < cap; ++i) {
        weights[i] = std::exp((sorted[i].score - sorted.front().score) /
                              static_cast<double>(config.temperature));
        total += weights[i];
    }
    int kept = 0;
    double kept_weight = 0.0;
    for (int i = 0; i < cap; ++i) {
        if (config.min_p > 0.0f && weights[i] < config.min_p * weights.front()) { break; }
        kept_weight += weights[i];
        ++kept;
        if (config.top_p < 1.0f && kept_weight >= config.top_p * total) { break; }
    }
    if (kept == 0) {
        kept = 1;
        kept_weight = weights.front();
    }
    std::vector<OracleCandidate> result;
    for (int i = 0; i < kept; ++i) {
        result.push_back({sorted[i].token, weights[i] / kept_weight});
    }
    return result;
}

struct JointOutcome {
    int accepted;
    int terminal_token;
    double probability;
    std::uint32_t observed = 0;
};

std::vector<JointOutcome> joint_outcome_oracle(
    const std::vector<std::vector<OracleCandidate>>& distributions,
    std::span<const std::int32_t> drafts, int extent) {
    std::vector<JointOutcome> result;
    double reach = 1.0;
    for (int column = 0; column <= extent; ++column) {
        double draft_probability = 0.0;
        for (const auto& candidate : distributions[column]) {
            if (column < extent && candidate.token == drafts[column]) {
                draft_probability = candidate.probability;
            } else {
                // Rejection: reach * (1-p(d)) * p(t)/(1-p(d)) = reach*p(t).
                // Bonus: reach*p(t). No simulated CPU draws are needed.
                if (reach * candidate.probability > 0.0) {
                    result.push_back({column, candidate.token, reach * candidate.probability});
                }
            }
        }
        reach *= draft_probability;
    }
    double total = 0.0;
    for (const auto& outcome : result) { total += outcome.probability; }
    if (std::abs(total - 1.0) > 1e-12) {
        throw std::runtime_error("speculative joint probability oracle did not normalize");
    }
    return result;
}

int check_joint_frequencies(const std::string& label, const std::vector<JointOutcome>& outcomes,
                            int draws) {
    // A two-sided binomial Chernoff/KL bound for each multinomial marginal,
    // followed by a union bound over at most 4096 tested bins in this executable.
    // Under independent uniform draws the family-wise false rejection bound is
    // <= 1e-6. This is a prespecified statistical criterion, not a numeric epsilon
    // fitted to observed GPU output. Seeds are distinct across every trial/lane.
    constexpr double family_error = 1e-6;
    constexpr int maximum_bins = 4096;
    const double cutoff = std::log(2.0 * maximum_bins / family_error);
    int failures = 0;
    double maximum_statistic = 0.0;
    for (const auto& outcome : outcomes) {
        const double p = outcome.probability;
        const double q = static_cast<double>(outcome.observed) / draws;
        double divergence = 0.0;
        if (p == 1.0) {
            divergence = q == 1.0 ? 0.0 : std::numeric_limits<double>::infinity();
        } else {
            if (q > 0.0) { divergence += q * std::log(q / p); }
            if (q < 1.0) { divergence += (1.0 - q) * std::log((1.0 - q) / (1.0 - p)); }
        }
        const double statistic = draws * std::max(0.0, divergence);
        maximum_statistic = std::max(maximum_statistic, statistic);
        if (statistic > cutoff) {
            std::cerr << label << ": A=" << outcome.accepted
                      << " terminal=" << outcome.terminal_token << " observed=" << q
                      << " expected=" << p << " N*KL=" << statistic
                      << " limit=" << cutoff << '\n';
            ++failures;
        }
    }
    std::cout << "    " << label << ": " << draws << " draws, " << outcomes.size()
              << " joint bins, max N*KL=" << maximum_statistic << " limit=" << cutoff << '\n';
    return failures;
}

int nondegenerate_sampling_case(int token_domain, int physical_rows, int batch, int iterations,
                                bool penalties) {
    constexpr int k = 3;
    constexpr int columns = k + 1;
    const std::string label = "speculative joint V=" + std::to_string(token_domain) +
                              " B=" + std::to_string(batch) + (penalties ? " penalties" : "");
    // Scatter the real-vocabulary support across distant sampler partitions, including
    // the final valid token. Padding receives enormous logits and must be ignored.
    const std::array<int, 5> ids = token_domain > 65537
        ? std::array<int, 5>{17, 7919, 65537, 200003, token_domain - 1}
        : std::array<int, 5>{3, 11, 29, 47, token_domain - 1};
    const float scores[columns][5] = {
        {1.5f, 0.4f, 0.0f, -0.5f, -2.0f},
        {0.6f, 1.7f, 0.1f, -0.2f, -1.7f},
        {1.6f, 0.9f, 0.0f, -0.3f, -2.0f},
        {0.1f, 0.5f, 1.2f, 0.8f, 0.4f},
    };
    std::vector<std::uint16_t> host_logits(
        static_cast<std::size_t>(physical_rows) * columns * batch, f32_to_bf16(-32.0f));
    std::vector<std::int32_t> host_drafts(static_cast<std::size_t>(k) * batch);
    std::vector<std::int32_t> host_extents(batch);
    std::vector<std::int32_t> host_targets(static_cast<std::size_t>(columns) * batch, ids[2]);
    std::vector<std::int32_t> initial_counts(
        penalties ? static_cast<std::size_t>(token_domain) * batch : 0, 0);
    std::vector<ops::SamplingConfig> configs(batch);
    std::vector<std::vector<JointOutcome>> expected(batch);
    std::size_t bin_count = 0;
    for (int row = 0; row < batch; ++row) {
        const int kind = row % 4;
        host_extents[row] = kind == 1 ? 1 : (kind == 2 ? 2 : k);
        host_drafts[k * row] = ids[0];
        host_drafts[k * row + 1] = ids[1];
        host_drafts[k * row + 2] = ids[0]; // repeated proposal exercises the penalty overlay
        auto& config = configs[row];
        config.temperature = 0.875f;
        config.top_k = 5;
        // Separate lanes make each filter observable on its own; the combined
        // lane alone can hide a broken filter when both remove the same token.
        config.top_p = kind == 1 ? 0.8f : (kind == 2 ? 1.0f : 0.96f);
        config.min_p = kind == 1 ? 0.0f : (kind == 2 ? 0.25f : 0.08f);
        if (penalties) {
            config.presence_penalty = 0.375f;
            config.frequency_penalty = 0.25f;
            initial_counts[static_cast<std::size_t>(row) * token_domain + ids[0]] = 1;
            initial_counts[static_cast<std::size_t>(row) * token_domain + ids[1]] = 2;
        }
        std::vector<std::vector<OracleCandidate>> distributions;
        double column_reach = 1.0;
        bool isolated_filter_changes_reachable_distribution = false;
        for (int column = 0; column < columns; ++column) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + column) * physical_rows;
            for (int i = 0; i < 5; ++i) {
                float value = scores[column][i];
                if (kind == 3 && column == 0) { value = i == 0 ? 20.0f : -20.0f; }
                if (kind == 3 && column == 1 && i == 1) { value = -20.0f; }
                host_logits[base + ids[i]] = f32_to_bf16(value);
            }
            for (int token = token_domain; token < physical_rows; ++token) {
                host_logits[base + token] = f32_to_bf16(100.0f);
            }
            const auto counts = penalties
                ? std::span<const std::int32_t>(initial_counts).subspan(
                      static_cast<std::size_t>(row) * token_domain, token_domain)
                : std::span<const std::int32_t>{};
            const auto column_logits =
                std::span<const std::uint16_t>(host_logits).subspan(base, token_domain);
            const auto prior_drafts =
                std::span<const std::int32_t>(host_drafts).subspan(k * row, column);
            distributions.push_back(
                verification_distribution(column_logits, config, counts, prior_drafts));
            const auto& distribution = distributions.back();
            if ((kind == 1 || kind == 2) && column <= host_extents[row] && column_reach > 0.0) {
                auto disabled = config;
                if (kind == 1) { disabled.top_p = 1.0f; }
                if (kind == 2) { disabled.min_p = 0.0f; }
                const auto without_filter =
                    verification_distribution(column_logits, disabled, counts, prior_drafts);
                const bool same = std::equal(
                    distribution.begin(), distribution.end(), without_filter.begin(),
                    without_filter.end(), [](const auto& a, const auto& b) {
                        return a.token == b.token && a.probability == b.probability;
                    });
                isolated_filter_changes_reachable_distribution |= !same;
            }
            if (column < host_extents[row]) {
                const auto proposal = std::find_if(
                    distribution.begin(), distribution.end(), [&](const auto& candidate) {
                        return candidate.token == host_drafts[k * row + column];
                    });
                column_reach *= proposal == distribution.end() ? 0.0 : proposal->probability;
            }
        }
        if ((kind == 1 || kind == 2) && !isolated_filter_changes_reachable_distribution) {
            throw std::runtime_error(
                kind == 1 ? "top_p fixture does not affect a reachable distribution"
                          : "min_p fixture does not affect a reachable distribution");
        }
        if (kind == 3 &&
            (distributions[0].size() != 1 || distributions[0][0].token != ids[0] ||
             std::any_of(distributions[1].begin(), distributions[1].end(),
                         [&](const auto& candidate) { return candidate.token == ids[1]; }))) {
            throw std::runtime_error("unit/zero proposal probability fixture is invalid");
        }
        expected[row] = joint_outcome_oracle(
            distributions, std::span<const std::int32_t>(host_drafts).subspan(k * row, k),
            host_extents[row]);
        bin_count += expected[row].size();
    }
    // Four calls below have <= 8 * 4 * 5 bins each, strictly below the union-bound budget.
    if (bin_count > 160) { throw std::runtime_error("joint-test bin budget exceeded"); }

    GuardedDeviceBuffer d_logits(host_logits.size() * sizeof(std::uint16_t));
    initialize(d_logits, host_logits);
    DeviceBuffer d_drafts = to_device(host_drafts);
    DeviceBuffer d_extents = to_device(host_extents);
    DeviceBuffer d_targets = to_device(host_targets);
    GuardedDeviceBuffer d_token_counts(initial_counts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_configs(configs.size() * sizeof(ops::SamplingConfig));
    // Disjoint views of one guarded output allocation permit one retirement copy
    // per trial; every field and every unused licensed slot is checked below.
    const int produced_offset = columns * batch;
    const int accepted_offset = produced_offset + batch;
    const int length_offset = accepted_offset + batch;
    const int anchor_offset = length_offset + batch;
    std::vector<std::int32_t> initial_io(anchor_offset + batch, -777);
    for (int row = 0; row < batch; ++row) { initial_io[length_offset + row] = 4093 + row * 97; }
    GuardedDeviceBuffer d_io(initial_io.size() * sizeof(std::int32_t));
    auto* io = static_cast<std::int32_t*>(d_io.data());
    for (int row = 0; row < batch; ++row) {
        configs[row].token_counts = penalties
            ? static_cast<std::int32_t*>(d_token_counts.data()) +
                  static_cast<std::size_t>(row) * token_domain
            : nullptr;
    }
    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits(d_logits.data(), DType::BF16, {physical_rows, columns, batch});
    Tensor drafts(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor licensed(io, DType::I32, {columns, batch});
    Tensor produced(io + produced_offset, DType::I32, {batch});
    Tensor accepted(io + accepted_offset, DType::I32, {batch});
    Tensor lengths(io + length_offset, DType::I32, {batch});
    Tensor anchors(io + anchor_offset, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k,
                                                                       batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    for (int trial = 0; trial < iterations; ++trial) {
        initialize(d_io, initial_io);
        initialize(d_token_counts, initial_counts);
        for (int row = 0; row < batch; ++row) {
            configs[row].seed = 0x7123000000000000ull +
                (static_cast<unsigned long long>(token_domain) << 24) +
                (static_cast<unsigned long long>(batch) << 20) +
                static_cast<unsigned long long>(trial * batch + row);
        }
        initialize(d_configs, configs);
        ops::speculative_accept_greedy_drafts(
            targets, logits, drafts, extents, lengths, anchors, licensed, produced, accepted,
            token_domain, static_cast<const ops::SamplingConfig*>(d_configs.data()), workspace,
            nullptr);
        const auto result = read<std::int32_t>(d_io, initial_io.size()); // synchronizes this trial
        auto expected_counts = initial_counts;
        for (int row = 0; row < batch; ++row) {
            const int a = result[accepted_offset + row];
            const int n = result[produced_offset + row];
            if (a < 0 || a > host_extents[row] || n != a + 1 ||
                result[length_offset + row] != initial_io[length_offset + row] + n) {
                std::cerr << label << ": invalid accepted count/length at trial=" << trial
                          << " row=" << row << '\n';
                return 1;
            }
            const int terminal = result[columns * row + a];
            if (terminal != result[anchor_offset + row]) {
                std::cerr << label << ": terminal/anchor mismatch\n";
                return 1;
            }
            auto& bins = expected[row];
            const auto found = std::find_if(bins.begin(), bins.end(), [&](const auto& outcome) {
                return outcome.accepted == a && outcome.terminal_token == terminal;
            });
            if (found == bins.end()) {
                std::cerr << label << ": impossible outcome A=" << a << " token=" << terminal
                          << " trial=" << trial << " row=" << row << '\n';
                return 1;
            }
            ++found->observed;
            for (int column = 0; column < columns; ++column) {
                const int token = result[columns * row + column];
                if ((column < a && token != host_drafts[k * row + column]) ||
                    (column >= n && token != 0)) {
                    std::cerr << label << ": licensed prefix/unused slot mismatch\n";
                    return 1;
                }
                if (penalties && column < n) {
                    ++expected_counts[static_cast<std::size_t>(row) * token_domain + token];
                }
            }
        }
        if (penalties && read<std::int32_t>(d_token_counts, initial_counts.size()) != expected_counts) {
            std::cerr << label << ": token counts include rejected/provisional tokens or lost repeats\n";
            return 1;
        }
    }
    int failures = 0;
    for (int row = 0; row < batch; ++row) {
        failures += check_joint_frequencies(label + " row=" + std::to_string(row), expected[row],
                                            iterations);
    }
    failures += verify_exact("joint logits unchanged",
                              read<std::uint16_t>(d_logits, host_logits.size()), host_logits);
    failures += verify_exact("joint drafts unchanged",
                              from_device<std::int32_t>(d_drafts, host_drafts.size()), host_drafts);
    failures += verify_exact("joint extents unchanged",
                              from_device<std::int32_t>(d_extents, host_extents.size()), host_extents);
    failures += verify_exact("joint targets unchanged",
                              from_device<std::int32_t>(d_targets, host_targets.size()), host_targets);
    std::vector<std::uint8_t> config_bytes(configs.size() * sizeof(ops::SamplingConfig));
    std::memcpy(config_bytes.data(), configs.data(), config_bytes.size());
    failures += verify_exact("joint configs unchanged",
                              read<std::uint8_t>(d_configs, config_bytes.size()), config_bytes);
    failures += d_io.verify_guards(label + " outputs");
    failures += d_logits.verify_guards(label + " logits");
    failures += d_configs.verify_guards(label + " configs");
    failures += d_token_counts.verify_guards(label + " token counts");
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {rows, 1});
    ops::speculative_select_accepted_hidden(hidden_tensor, accepted, out, nullptr);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures = verify_exact((label + " output").c_str(),
                                read<std::uint16_t>(d_out, expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    initialize(d_proposals, proposals);
    Tensor proposal_tensor(d_proposals.data(), DType::I32, {token_count});
    ops::proposal_remap_token_ids(proposal_tensor, static_cast<const std::int32_t*>(d_map.p),
                                  map_size, nullptr);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures            = verify_exact((label + " in-place output").c_str(),
                                           read<std::int32_t>(d_proposals, proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "speculative_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    if (k15 == 0 || k15 != ops::sampling_workspace_capacity_bytes(257, 16, 16) ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 16, 16, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 16, 1, 1) != k15 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not close over K+1 sampling columns\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (const int k : {1, 5, 15}) failures += prepare_verify_case(k);
    failures += greedy_accept_case(1, 0);
    failures += greedy_accept_case(5, 2);
    failures += greedy_accept_case(5, 5);
    failures += greedy_accept_case(15, 7, 257);
    failures += deterministic_sampling_case();
    failures += batched_sampling_workspace_stride_case();
    failures += nondegenerate_sampling_case(64, 80, 1, 8192, false);
    failures += nondegenerate_sampling_case(257, 272, 2, 4096, true);
    failures += nondegenerate_sampling_case(128, 144, 8, 2048, true);
    failures += nondegenerate_sampling_case(248077, 248320, 8, 2048, false);
    failures += select_hidden_case(5120, 6, 0);
    failures += select_hidden_case(5120, 6, 5);
    failures += select_hidden_case(2048, 16, 7);
    failures += remap_case(1);
    failures += remap_case(15);
    failures += remap_case(120);

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
