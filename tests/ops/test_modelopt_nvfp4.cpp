// Independent represented-input oracle for the calibrated ModelOpt NVFP4 route.
// Source codes and scale words are decoded logically; the FP64 dot never copies
// a CUDA reduction tree or its staging precision. Samples span physical tile seams.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include <array>
#include <cstring>
#include <iomanip>
#include <set>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr float weight_multiplier = 0.017932971939444542F;
constexpr float input_multiplier = 0.00371937220916152F;
constexpr auto policy = ops::LinearPolicy::CalibratedA4;

double fp4(unsigned code) {
    constexpr double values[8]{0, .5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8U ? -1 : 1) * values[code & 7U];
}
double fp8(unsigned code) {
    const unsigned exponent = (code >> 3) & 15U;
    const unsigned fraction = code & 7U;
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -9)
        : std::ldexp(1.0 + double(fraction) / 8, int(exponent) - 7);
    return code & 128U ? -magnitude : magnitude;
}
// Enumerating the finite format is intentionally independent of GPU conversion instructions.
unsigned nearest_positive(double value, bool four_bit) {
    const unsigned last = four_bit ? 7U : 126U;
    unsigned best = 0;
    double error = std::abs(value);
    for (unsigned code = 1; code <= last; ++code) {
        const double decoded = four_bit ? fp4(code) : fp8(code);
        const double candidate = std::abs(value - decoded);
        if (candidate < error || (candidate == error && (code & 1U) == 0)) {
            best = code;
            error = candidate;
        }
    }
    return best;
}
unsigned source_code(int row, int column) {
    const unsigned bits = unsigned(row + 7) * 2654435761U ^ unsigned(column + 11) * 2246822519U;
    return (bits ^ (bits >> 13)) & 15U;
}
unsigned source_scale(int row, int group) {
    return 32U + ((unsigned(row) * 17U + unsigned(group) * 13U) % 31U);
}
std::size_t scale_index(int n, int k, int row, int group) {
    (void)n;
    return (std::size_t(row / 128) * (k / 64) + group / 4) * 512
        + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}
struct Fixture {
    std::vector<std::uint8_t> bytes;
    std::size_t scale_offset;
    int n, k;
    Fixture(int rows, int columns) : scale_offset(std::size_t(rows) * columns / 2), n(rows), k(columns) {
        bytes.resize(scale_offset + std::size_t(n) * k / 16 + 4);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < k; col += 2) {
                bytes[std::size_t(row) * k / 2 + col / 2] = std::uint8_t(
                    source_code(row, col) | (source_code(row, col + 1) << 4));
            }
            for (int group = 0; group < k / 16; ++group) {
                bytes[scale_offset + scale_index(n, k, row, group)] = std::uint8_t(source_scale(row, group));
            }
        }
        std::memcpy(bytes.data() + bytes.size() - 4, &weight_multiplier, 4);
    }
    Weight view(void* data) const {
        Weight w{};
        w.qtype = QType::NVFP4_F32M;
        w.layout = QuantLayout::BlockScaleK16M128x4Multiplier;
        w.scale_dtype = DType::FP8_E4M3FN;
        w.ndim = 2;
        w.n = n; w.k = k; w.group = w.group_size = 16;
        w.shape[0] = w.padded_shape[0] = n;
        w.shape[1] = w.padded_shape[1] = k;
        w.payload = w.qdata = data;
        w.payload_bytes = bytes.size();
        w.scales = static_cast<std::uint8_t*>(data) + scale_offset;
        w.weight_scale_multiplier = weight_multiplier;
        w.input_scale_multiplier = input_multiplier;
        return w;
    }
};

std::vector<float> input(int k, int t) {
    std::vector<float> x(std::size_t(k) * t);
    fill_uniform(x, 3121U + unsigned(t), -1.0F, 1.0F);
    for (int token = 0; token < t; ++token) {
        for (int col = 0; col < 16; ++col) x[std::size_t(token)*k+col] = 0;
        for (int col = 16; col < 32; ++col) x[std::size_t(token)*k+col] = col % 2 ? 1e-7F : -1e-7F;
        for (int col = 32; col < 48; ++col) x[std::size_t(token)*k+col] = float(col - 40) * 3.0F;
    }
    round_to_bf16(x);
    return x;
}
struct EncodedInput { std::vector<double> values; std::vector<std::uint8_t> codes, scales; };
EncodedInput oracle_input(const std::vector<float>& x) {
    EncodedInput out;
    out.values.resize(x.size()); out.codes.resize(x.size()/2); out.scales.resize(x.size()/16);
    for (std::size_t begin = 0; begin < x.size(); begin += 16) {
        double maximum = 0;
        for (int j=0;j<16;++j) maximum = std::max(maximum, std::abs(double(x[begin+j])));
        const unsigned s = nearest_positive(std::min(448.0, maximum / (6.0 * double(input_multiplier))), false);
        out.scales[begin/16] = std::uint8_t(s);
        const double scale = fp8(s) * double(input_multiplier);
        for (int j=0;j<16;++j) {
            const unsigned code = scale == 0 ? 0 : nearest_positive(std::abs(double(x[begin+j]))/scale, true)
                | (std::signbit(x[begin+j]) ? 8U : 0U);
            out.values[begin+j] = fp4(code) * scale;
            out.codes[(begin+j)/2] |= std::uint8_t(code << ((j&1)*4));
        }
    }
    return out;
}
std::pair<double,double> dot(int row,int k,const double* x) {
    double result=0, magnitude=0;
    for(int col=0;col<k;++col) {
        const double w=fp4(source_code(row,col))*fp8(source_scale(row,col/16))*double(weight_multiplier);
        const double term=w*x[col]; result+=term; magnitude+=std::abs(term);
    }
    return {result,magnitude};
}
enum class Kind { Linear, Add, SwiGlu };
int run(int n,int k,int t,Kind kind,cudaStream_t stream) {
    Fixture fixture(n,k);
    auto weights=to_device(fixture.bytes);
    const auto w=fixture.view(weights.p);
    auto x=input(k,t); const auto oracle=oracle_input(x);
    auto dx=to_device_bf16(x);
    const int out_n=kind==Kind::SwiGlu ? n/2 : n;
    std::vector<float> residual(std::size_t(out_n)*t, .125F);
    auto dy=to_device_bf16(residual);
    Tensor tx(dx.p,DType::BF16,{k,t}), ty(dy.p,DType::BF16,{out_n,t});
    std::size_t capacity=kind==Kind::Linear ? ops::linear_workspace_capacity_bytes(w.qtype,n,k,policy,t,t)
        : kind==Kind::Add ? ops::linear_add_workspace_capacity_bytes(w.qtype,n,k,policy,t,t)
        : ops::linear_swiglu_workspace_capacity_bytes(w.qtype,n,k,policy,t,t);
    WorkspaceArena workspace(capacity);
    cuda_synchronize();
    if(kind==Kind::Linear) ops::linear(tx,w,ty,policy,workspace,stream);
    else if(kind==Kind::Add) ops::linear_add(tx,w,ty,policy,workspace,stream);
    else ops::linear_swiglu(tx,w,ty,policy,workspace,stream);
    cuda_synchronize(stream);
    const auto result=from_device_bf16(dy,residual.size());
    const std::set<int> rows{0,1,31,32,63,64,127,128,out_n/2-1,out_n/2,out_n-2,out_n-1};
    const std::set<int> tokens{0,t/2,t-1};
    int failures=0; double worst=0;
    for(int token:tokens) for(int row:rows) {
        auto [reference,magnitude]=dot(row,k,oracle.values.data()+std::size_t(token)*k);
        double limit=0.006*std::abs(reference)+2e-6*magnitude+2e-5;
        if(kind==Kind::Add) { reference+=.125; limit+=0.001; }
        if(kind==Kind::SwiGlu) {
            const auto [up,up_magnitude]=dot(row+n/2,k,oracle.values.data()+std::size_t(token)*k);
            reference=reference/(1+std::exp(-reference))*up;
            limit=.025*std::abs(reference)+1e-5*(magnitude+up_magnitude)+2e-5;
        }
        const double actual=result[std::size_t(token)*out_n+row];
        const double error=std::abs(actual-reference);
        worst=std::max(worst,error/std::max(limit,1e-12));
        if(!std::isfinite(actual)||error>limit) {
            if(failures<3) std::cerr<<" mismatch r="<<row<<" t="<<token<<" got="<<actual<<" expected="<<reference<<" limit="<<limit<<'\n';
            ++failures;
        }
    }
    // Check the explicit activation quantize boundary separately, including zero,
    // underflow, saturation and packed nibble order, against the independent codec.
    const auto qbytes=ops::detail::nvfp4_w4a4_workspace_capacity_bytes(t,k);
    WorkspaceArena quant_workspace(qbytes);
    auto scratch=ops::detail::allocate_nvfp4_w4a4_workspace(quant_workspace,t,k);
    ops::detail::launch_nvfp4_w4a4_quantize(tx,w,scratch,stream);
    cuda_synchronize(stream);
    const auto codes=from_device<std::uint8_t>(scratch.codes,oracle.codes.size());
    const auto scales=from_device<std::uint8_t>(scratch.scales,oracle.scales.size());
    if(codes!=oracle.codes||scales!=oracle.scales) { std::cerr<<" activation codec differs from oracle\n"; ++failures; }
    std::cout<<"NVFP4_F32M "<<n<<'x'<<k<<" T="<<t<<" kind="<<int(kind)<<" worst/limit="<<worst<<" failures="<<failures<<'\n';
    return failures;
}
}
int main() {
    if(cuda_unavailable()) return 77;
    try {
        int devices=0; cuda_check(cudaGetDeviceCount(&devices),"device count");
        int failures=0;
        for(int device=0;device<std::min(2,devices);++device) {
            cuda_check(cudaSetDevice(device),"set device");
            cudaStream_t stream{}; cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
            std::cout<<"device="<<device<<'\n';
            for(int t:{1,4,17,1024}) {
                failures+=run(17408,5120,t,Kind::Linear,stream);
                failures+=run(5120,8704,t,Kind::Linear,stream);
                failures+=run(34816,5120,t,Kind::SwiGlu,stream);
                failures+=run(5120,17408,t,Kind::Add,stream);
            }
            for(int n:{248320,124160,131072,65536}) for(int t:{1,4}) failures+=run(n,5120,t,Kind::Linear,stream);
            cuda_check(cudaStreamDestroy(stream),"destroy stream");
        }
        return failures ? 1 : 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
