// Independent ModelOpt FP8 contract qualification: original FP32 calibration,
// exact E4M3 activation quantization and complete-K FP64 dots at sampled output
// coordinates. Sampling spans every output tile and token-tile seams; every output
// is checked for finiteness, and every input/weight byte and allocation guard survives.
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/op_tester.h"

#include <array>
#include <memory>
#include <set>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr auto format = QType::FP8_E4M3FN_ROW_F32S;
constexpr auto policy = ops::LinearPolicy::CalibratedA8;
constexpr float input_scale = 0.00371937220916152F;

double decode(unsigned code) {
    const unsigned exponent = (code >> 3) & 15U, fraction = code & 7U;
    if (exponent == 15 && fraction == 7) throw std::runtime_error("oracle NaN code");
    const double value = exponent == 0 ? std::ldexp(double(fraction), -9)
        : std::ldexp(1.0 + double(fraction) / 8, int(exponent) - 7);
    return code & 128U ? -value : value;
}
// Search the format, independent of CUDA conversion intrinsics. The quotient is
// explicitly FP32 per the calibrated activation codec; E4M3 rounding is nearest-even.
std::uint8_t encode(float represented, float scale) {
    const float quotient = represented / scale;
    const double value = std::min(448.0, std::abs(double(quotient)));
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        const unsigned mid = (lo + hi) / 2;
        if (decode(mid) < value) lo = mid + 1; else hi = mid;
    }
    unsigned best = lo;
    if (lo > 0) {
        const double below = value - decode(lo - 1), above = decode(lo) - value;
        if (below < above || (below == above && ((lo - 1) & 1U) == 0)) best = lo - 1;
    }
    return std::uint8_t(best | (std::signbit(represented) ? 128U : 0U));
}
unsigned mix(unsigned value) {
    value ^= value >> 16; value *= 0x7feb352dU;
    value ^= value >> 15; value *= 0x846ca68bU;
    return value ^ (value >> 16);
}
std::vector<std::uint16_t> make_input(int k, int t) {
    std::vector<std::uint16_t> x(std::size_t(k) * t);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int signed_value = int(mix(unsigned(i) + 7181U) % 2049U) - 1024;
        x[i] = f32_to_bf16(float(signed_value) / 1024.0F);
    }
    constexpr float edges[]{1.0F, 0.0F, -0.0F, 1e-7F, -1e-7F, 64.0F, -64.0F,
                            0.00390625F, 0.004150390625F, 0.00830078125F};
    for (int token = 0; token < t; ++token)
        for (int j = 0; j < int(std::size(edges)); ++j)
            x[std::size_t(token) * k + j] = f32_to_bf16(edges[j]);
    return x;
}
struct EncodedInput {
    std::vector<std::uint8_t> codes;
    std::vector<double> represented;
    explicit EncodedInput(const std::vector<std::uint16_t>& input) {
        // The table only memoizes this test's independent codec, never production outputs.
        std::array<std::uint8_t, 65536> table{};
        for (unsigned bits = 0; bits < table.size(); ++bits) {
            const float value = bf16_to_f32(std::uint16_t(bits));
            if (std::isfinite(value)) table[bits] = encode(value, input_scale);
        }
        codes.resize(input.size()); represented.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            codes[i] = table[input[i]];
            represented[i] = decode(codes[i]) * double(input_scale);
        }
    }
};
enum class Kind { Linear, Add, Attention, Gdn, SwiGlu };
struct Block { int rows, origin; };
std::vector<Block> output_blocks(Kind kind, int n) {
    if (kind == Kind::Attention) return {{3072, 0}, {3072, 3584}, {512, 3072}, {512, 6656}};
    if (kind == Kind::Gdn) return {{5120, 0}, {3072, 5120}};
    return {{kind == Kind::SwiGlu ? n / 2 : n, 0}};
}
std::size_t capacity(Kind kind, int n, int k, int t) {
    switch (kind) {
    case Kind::Attention: return ops::attn_input_proj_column_parallel_workspace_capacity_bytes(format, policy, t, t);
    case Kind::Gdn: return ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(format, policy, t, t);
    case Kind::SwiGlu: return ops::linear_swiglu_column_parallel_workspace_capacity_bytes(format, policy, t, t);
    case Kind::Add: return ops::linear_add_workspace_capacity_bytes(format, n, k, policy, t, t);
    case Kind::Linear: return ops::linear_workspace_capacity_bytes(format, n, k, policy, t, t);
    }
    throw std::logic_error("unknown kind");
}
struct Fixture {
    int device, n, k, t;
    Kind kind;
    cudaStream_t stream;
    std::size_t scale_offset;
    std::vector<std::uint8_t> weight_bytes;
    std::vector<std::uint16_t> input;
    EncodedInput encoded;
    GuardedDeviceBuffer weight_storage, input_storage, scratch_storage;
    WorkspaceArena workspace;
    std::vector<Block> blocks;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> output_storage;
    std::vector<Tensor> output;
    Tensor x;
    Weight w{};
    Fixture(int device_id, cudaStream_t selected_stream, Kind selected_kind, int rows, int columns, int tokens)
        : device(device_id), n(rows), k(columns), t(tokens), kind(selected_kind), stream(selected_stream),
          scale_offset((std::size_t(n) * k + 255) & ~std::size_t(255)),
          weight_bytes(scale_offset + std::size_t(n) * 4), input(make_input(k, t)), encoded(input),
          weight_storage(weight_bytes.size()), input_storage(input.size() * 2),
          scratch_storage(capacity(kind, n, k, t)),
          workspace(DeviceSpan{scratch_storage.data(), scratch_storage.bytes()}), blocks(output_blocks(kind, n)),
          x(input_storage.data(), DType::BF16, {k, t}) {
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < k; ++col) {
                const unsigned bits = mix(unsigned(row) * 65537U + unsigned(col) + unsigned(device) * 113U);
                // Bounded, nondegenerate signed E4M3 weights; the first rows are
                // sparse calibration witnesses while the remaining matrix is dense.
                weight_bytes[std::size_t(row) * k + col] = std::uint8_t(row < 8
                    ? (col == 0 ? 56U : 0U) : (24U + bits % 41U) | ((bits >> 17) & 128U));
            }
            const float scale = row < 8 ? 0.001702808542177081F + float(row) * 0.0000000137F
                : 0.001713579F + float(mix(unsigned(row) + unsigned(device) * 191U) % 97U) * 0.000003173F;
            std::memcpy(weight_bytes.data() + scale_offset + std::size_t(row) * 4, &scale, 4);
        }
        weight_storage.copy_from_host(weight_bytes.data(), weight_bytes.size());
        input_storage.copy_from_host(input.data(), input.size() * 2);
        w.qtype = format; w.layout = QuantLayout::RowScaleF32; w.scale_dtype = DType::FP32;
        w.n = n; w.k = k; w.ndim = 2; w.group = k; w.group_size = unsigned(k);
        w.shape[0] = w.padded_shape[0] = n; w.shape[1] = w.padded_shape[1] = k;
        w.scale_ne[0] = n; w.scale_nb[0] = 4;
        w.scale_nb[1] = w.scale_nb[2] = w.scale_nb[3] = std::int64_t(n) * 4;
        w.payload = w.qdata = weight_storage.data(); w.payload_bytes = weight_bytes.size();
        w.scales = static_cast<std::uint8_t*>(weight_storage.data()) + scale_offset;
        w.input_scale_multiplier = input_scale;
        for (Block block : blocks) {
            output_storage.push_back(std::make_unique<GuardedDeviceBuffer>(std::size_t(block.rows) * t * 2));
            output.emplace_back(output_storage.back()->data(), DType::BF16, std::initializer_list<std::int32_t>{block.rows, t});
        }
        reset_outputs();
        cuda_synchronize(); // retire default-stream fixture uploads before the nonblocking stream
    }
    ~Fixture() {
        (void)cudaSetDevice(device);
        (void)cudaStreamSynchronize(stream);
    }
    void reset_outputs() {
        for (auto& buffer : output_storage) {
            const std::vector<std::uint16_t> initial(buffer->bytes() / 2,
                kind == Kind::Add ? f32_to_bf16(.125F) : std::uint16_t(0x7fc1));
            buffer->copy_from_host(initial.data(), buffer->bytes());
        }
    }
    std::pair<double, double> dot(int row, int token) const {
        float scale; std::memcpy(&scale, weight_bytes.data() + scale_offset + std::size_t(row) * 4, 4);
        double sum = 0, magnitude = 0;
        for (int col = 0; col < k; ++col) {
            const double weight = decode(weight_bytes[std::size_t(row) * k + col]) * double(scale);
            const double term = weight * encoded.represented[std::size_t(token) * k + col];
            sum += term; magnitude += std::abs(term);
        }
        return {sum, magnitude};
    }
    int verify() const {
        int failures = 0;
        double worst = 0;
        std::set<int> tokens;
        if (t <= 4) for (int token = 0; token < t; ++token) tokens.insert(token);
        else tokens = {0, 1, 15, 16, 31, 32, 63, 64, t / 2, t - 1};
        int calibration_witnesses = 0;
        for (std::size_t section = 0; section < blocks.size(); ++section) {
            const Block block = blocks[section];
            const auto result = from_device_bf16(output[section].data, std::size_t(block.rows) * t);
            if (!std::all_of(result.begin(), result.end(), [](double v) { return std::isfinite(v); })) {
                std::cerr << "unwritten or non-finite output\n"; ++failures;
            }
            std::set<int> rows{0, 1, 7, 31, 32, 63, 64, block.rows - 2, block.rows - 1};
            for (int row = 127; row < block.rows; row += 128) { rows.insert(row); if (row + 1 < block.rows) rows.insert(row + 1); }
            for (int token : tokens) if (token < t) for (int row : rows) if (row < block.rows) {
                auto [reference, magnitude] = dot(block.origin + row, token);
                double limit = .005 * std::abs(reference) + 3e-7 * magnitude + 1e-6;
                if (kind == Kind::Add) { reference += .125; limit += .0005; }
                if (kind == Kind::SwiGlu) {
                    const auto [up, up_magnitude] = dot(row + n / 2, token);
                    reference = reference / (1 + std::exp(-reference)) * up;
                    // The inherited vector epilogue privately materializes gate/up
                    // as BF16. The oracle does not copy those staging casts.
                    limit = .02 * std::abs(reference) + 2e-6 * (magnitude + up_magnitude) + 1e-6;
                }
                const double actual = result[std::size_t(token) * block.rows + row];
                if ((kind == Kind::Linear || kind == Kind::Attention || kind == Kind::Gdn) &&
                    block.origin + row < 8) {
                    // A single nonzero product has no reduction uncertainty. This
                    // exact output witness makes silently rounding row scales to
                    // BF16 observable, beyond the dot-product tolerance below.
                    const auto expected = f32_to_bf16(float(reference));
                    float stored; std::memcpy(&stored, weight_bytes.data() + scale_offset + std::size_t(row) * 4, 4);
                    const double rounded = encoded.represented[std::size_t(token) * k]
                        * double(bf16_to_f32(f32_to_bf16(stored)));
                    if (expected != f32_to_bf16(float(rounded))) ++calibration_witnesses;
                    if (f32_to_bf16(float(actual)) != expected) {
                        std::cerr << "single-product FP32 row-scale witness failed\n"; ++failures;
                    }
                }
                const double error = std::abs(actual - reference);
                worst = std::max(worst, error / limit);
                if (!std::isfinite(actual) || error > limit) {
                    if (failures < 3) std::cerr << "section=" << section << " r=" << row << " t=" << token
                        << " got=" << actual << " expected=" << reference << " limit=" << limit << '\n';
                    ++failures;
                }
            }
            failures += output_storage[section]->verify_guards("FP8 output");
        }
        if ((kind == Kind::Linear || kind == Kind::Attention || kind == Kind::Gdn) && calibration_witnesses == 0) {
            std::cerr << "fixture has no observable FP32 row-scale witness\n"; ++failures;
        }
        failures += weight_storage.verify_guards("FP8 weights") + input_storage.verify_guards("FP8 input")
                    + scratch_storage.verify_guards("FP8 workspace");
        failures += verify_exact("FP8 weight preservation", from_device<std::uint8_t>(w.payload, weight_bytes.size()), weight_bytes);
        failures += verify_exact("FP8 input preservation", from_device<std::uint16_t>(x.data, input.size()), input);
        if (workspace.used() != 0) { std::cerr << "FP8 workspace scope leaked\n"; ++failures; }
        std::cout << "FP8_F32S device=" << device << " kind=" << int(kind) << " N=" << n << " K=" << k
                  << " T=" << t << " sampled-full-K worst/limit=" << worst << " failures=" << failures << '\n';
        return failures;
    }
    int verify_codec() {
        GuardedDeviceBuffer storage(ops::detail::fp8_a8_workspace_capacity_bytes(t, k));
        WorkspaceArena arena(DeviceSpan{storage.data(), storage.bytes()});
        auto scratch = ops::detail::allocate_fp8_a8_workspace(arena, t, k);
        cuda_synchronize();
        ops::detail::launch_fp8_a8_quantize(x, w, scratch, stream);
        cuda_synchronize(stream);
        int failures = verify_exact("calibrated activation codes", from_device<std::uint8_t>(scratch.codes, input.size()), encoded.codes);
        const auto scales = from_device<float>(scratch.scales, t);
        if (std::any_of(scales.begin(), scales.end(), [](float v) { return std::memcmp(&v, &input_scale, 4) != 0; })) {
            std::cerr << "input multiplier bits were changed\n"; ++failures;
        }
        return failures + storage.verify_guards("activation codec scratch");
    }
};
void issue(std::array<std::unique_ptr<Fixture>, 2>& fixtures, const ExecutionContext& ec, Kind kind) {
    const std::array<Tensor, 2> x{fixtures[0]->x, fixtures[1]->x};
    const std::array<Weight, 2> w{fixtures[0]->w, fixtures[1]->w};
    const std::array<WorkspaceArena*, 2> ws{&fixtures[0]->workspace, &fixtures[1]->workspace};
    auto outputs = [&](int section) { return std::array<Tensor, 2>{fixtures[0]->output[section], fixtures[1]->output[section]}; };
    if (kind == Kind::Attention) ops::attn_input_proj_column_parallel(x, w, outputs(0), outputs(1), outputs(2), outputs(3), policy, ws, ec);
    else if (kind == Kind::Gdn) ops::gdn_input_proj_column_parallel(x, w, outputs(0), outputs(1), policy, ws, ec);
    else if (kind == Kind::SwiGlu) ops::linear_swiglu_column_parallel(x, w, outputs(0), policy, ws, ec);
    else for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select rank");
        auto& f = *fixtures[rank];
        if (kind == Kind::Linear) ops::linear(f.x, f.w, f.output[0], policy, f.workspace, f.stream);
        else ops::linear_add(f.x, f.w, f.output[0], policy, f.workspace, f.stream);
    }
}
int run_case(const ExecutionContext& ec, Kind kind, int n, int k, int t) {
    std::array<std::unique_ptr<Fixture>, 2> fixtures;
    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "fixture device");
        fixtures[rank] = std::make_unique<Fixture>(ec.dev[rank]->device, ec.dev[rank]->stream, kind, n, k, t);
    }
    issue(fixtures, ec, kind);
    int failures = 0;
    for (auto& f : fixtures) {
        cuda_check(cudaSetDevice(f->device), "verify device"); cuda_synchronize(f->stream);
        failures += f->verify();
        if (kind == Kind::Linear) failures += f->verify_codec();
    }
    return failures;
}
int admission() {
    int failures = 0;
    for (auto selected : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::CalibratedA4}) {
        bool rejected = false;
        try { (void)ops::linear_workspace_capacity_bytes(format, 5120, 3072, selected, 1, 4); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) ++failures;
    }
    bool rejected = false;
    try { (void)ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, policy, 1, 4); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) ++failures;
    if (ops::linear_workspace_capacity_bytes(format, 5120, 3072, policy, 1, 1) == 0) ++failures;
    return failures;
}
} // namespace
int main() {
    if (cuda_unavailable()) return 77;
    try {
        int count = 0; cuda_check(cudaGetDeviceCount(&count), "device count");
        if (count < 2) return 77;
        const ExecutionContext ec({0, 1});
        int failures = admission();
        for (int t : {1, 4, 1024}) {
            failures += run_case(ec, Kind::Linear, 5120, 3072, t);
            failures += run_case(ec, Kind::Add, 5120, 3072, t);
            failures += run_case(ec, Kind::Attention, 7168, 5120, t);
            failures += run_case(ec, Kind::Gdn, 8192, 5120, t);
            failures += run_case(ec, Kind::SwiGlu, 17408, 5120, t);
        }
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
