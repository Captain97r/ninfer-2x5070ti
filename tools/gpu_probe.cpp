#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Build/run via tools/run_gpu_probe.ps1. No kernels or model downloads required.
// Each GPU allocates one 16 MiB buffer; a portable pinned host buffer stages copies.
namespace {
void check(cudaError_t result, const char* expression) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) check((call), #call)
using Clock = std::chrono::steady_clock;
constexpr size_t kMaximumBytes = 16 * 1024 * 1024;

std::string quote(const char* text) {
    std::ostringstream out;
    out << '"';
    for (const char* p = text; *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\';
        out << *p;
    }
    out << '"';
    return out.str();
}

struct Device {
    int ordinal;
    cudaDeviceProp properties{};
    size_t freeBytes{}, totalBytes{};
    void* buffer{};
    cudaStream_t stream{};
};

// This is deliberately a serial staging baseline: no pipeline/double buffering.
// Timings include both GPU copies, device switches, and stream synchronization.
void stagedCopy(Device& source, Device& target, void* stage, size_t bytes) {
    CUDA(cudaSetDevice(source.ordinal));
    CUDA(cudaMemcpyAsync(stage, source.buffer, bytes, cudaMemcpyDeviceToHost, source.stream));
    CUDA(cudaStreamSynchronize(source.stream));
    CUDA(cudaSetDevice(target.ordinal));
    CUDA(cudaMemcpyAsync(target.buffer, stage, bytes, cudaMemcpyHostToDevice, target.stream));
    CUDA(cudaStreamSynchronize(target.stream));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
}  // namespace

int main() {
    try {
        int count{}, runtimeVersion{}, driverVersion{};
        CUDA(cudaGetDeviceCount(&count));
        CUDA(cudaRuntimeGetVersion(&runtimeVersion));
        CUDA(cudaDriverGetVersion(&driverVersion));
        if (count < 2) throw std::runtime_error("At least two CUDA devices are required.");
        std::vector<Device> devices(count);
        for (int i = 0; i < count; ++i) {
            devices[i].ordinal = i;
            CUDA(cudaSetDevice(i));
            CUDA(cudaGetDeviceProperties(&devices[i].properties, i));
            CUDA(cudaMemGetInfo(&devices[i].freeBytes, &devices[i].totalBytes));
        }
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "{\n  \"schema_version\": 1,\n  \"cuda_runtime_version\": " << runtimeVersion
            << ",\n  \"cuda_driver_api_version\": " << driverVersion << ",\n  \"devices\": [\n";
        for (int i = 0; i < count; ++i) {
            const auto& d = devices[i];
            const auto& p = d.properties;
            int sharedOptin{}, memoryBusWidth{}, l2Bytes{};
            CUDA(cudaDeviceGetAttribute(&sharedOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, i));
            CUDA(cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, i));
            CUDA(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, i));
            out << "    {\"ordinal\": " << i << ", \"name\": " << quote(p.name)
                << ", \"compute_capability\": " << quote((std::to_string(p.major) + "." + std::to_string(p.minor)).c_str())
                << ", \"sm_count\": " << p.multiProcessorCount
                << ", \"warp_size\": " << p.warpSize
                << ", \"shared_memory_per_block_bytes\": " << p.sharedMemPerBlock
                << ", \"shared_memory_per_block_optin_bytes\": " << sharedOptin
                << ", \"shared_memory_per_sm_bytes\": " << p.sharedMemPerMultiprocessor
                << ", \"registers_per_block\": " << p.regsPerBlock
                << ", \"registers_per_sm\": " << p.regsPerMultiprocessor
                << ", \"max_threads_per_sm\": " << p.maxThreadsPerMultiProcessor
                << ", \"memory_bus_width_bits\": " << memoryBusWidth
                << ", \"l2_cache_bytes\": " << l2Bytes
                << ", \"free_memory_before_probe_buffers_bytes\": " << d.freeBytes
                << ", \"total_memory_bytes\": " << d.totalBytes
                << ", \"pci_domain_id\": " << p.pciDomainID
                << ", \"pci_bus_id\": " << p.pciBusID
                << ", \"pci_device_id\": " << p.pciDeviceID
                << ", \"tcc_driver\": " << (p.tccDriver ? "true" : "false") << "}"
                << (i + 1 < count ? "," : "") << "\n";
        }
        out << "  ],\n  \"peer_access\": [\n";
        bool first = true;
        for (int from = 0; from < count; ++from) for (int to = 0; to < count; ++to) {
            if (from == to) continue;
            int canAccess{};
            CUDA(cudaDeviceCanAccessPeer(&canAccess, from, to));
            if (!first) out << ",\n";
            first = false;
            out << "    {\"from\": " << from << ", \"to\": " << to
                << ", \"cuda_device_can_access_peer\": " << (canAccess ? "true" : "false") << "}";
        }
        out << "\n  ],\n  \"staging\": {\"host_allocation\": \"cudaHostAllocPortable\", "
            << "\"gpu_buffer_bytes_per_device\": " << kMaximumBytes
            << ", \"warmup_iterations\": 8, \"method\": "
            << "\"Serial D2H then H2D; per-iteration steady_clock includes cudaSetDevice and both cudaStreamSynchronize calls; no overlap.\"},\n"
            << "  \"host_staged_transfers\": [\n";
        void* stage{};
        CUDA(cudaHostAlloc(&stage, kMaximumBytes, cudaHostAllocPortable));
        std::vector<uint8_t> expected(kMaximumBytes), actual(kMaximumBytes);
        for (size_t i = 0; i < kMaximumBytes; ++i) expected[i] = static_cast<uint8_t>((i * 17 + i / 251 + 29) & 255);
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaMalloc(&devices[i].buffer, kMaximumBytes));
            CUDA(cudaStreamCreateWithFlags(&devices[i].stream, cudaStreamNonBlocking));
        }
        first = true;
        for (int direction = 0; direction < 2; ++direction) {
            Device& source = devices[direction];
            Device& target = devices[1 - direction];
            CUDA(cudaSetDevice(source.ordinal));
            CUDA(cudaMemcpy(source.buffer, expected.data(), kMaximumBytes, cudaMemcpyHostToDevice));
            CUDA(cudaDeviceSynchronize());
            CUDA(cudaSetDevice(target.ordinal));
            CUDA(cudaMemset(target.buffer, 0, kMaximumBytes));
            CUDA(cudaDeviceSynchronize());
            for (size_t bytes : {size_t(8 * 1024), size_t(10 * 1024), size_t(64 * 1024), size_t(1024 * 1024), kMaximumBytes}) {
                for (int i = 0; i < 8; ++i) stagedCopy(source, target, stage, bytes);
                const int iterations = bytes < kMaximumBytes ? 100 : 40;
                std::vector<double> samples;
                for (int i = 0; i < iterations; ++i) {
                    const auto start = Clock::now();
                    stagedCopy(source, target, stage, bytes);
                    samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
                }
                CUDA(cudaSetDevice(target.ordinal));
                CUDA(cudaMemcpy(actual.data(), target.buffer, bytes, cudaMemcpyDeviceToHost));
                if (std::memcmp(expected.data(), actual.data(), bytes) != 0)
                    throw std::runtime_error("Host-staged copy correctness verification failed.");
                double sum{};
                for (double us : samples) sum += us;
                const double average = sum / iterations;
                std::sort(samples.begin(), samples.end());
                if (!first) out << ",\n";
                first = false;
                out << "    {\"from\": " << source.ordinal << ", \"to\": " << target.ordinal
                    << ", \"bytes\": " << bytes << ", \"iterations\": " << iterations
                    << ", \"mean_us\": " << average << ", \"median_us\": " << median(samples)
                    << ", \"p95_us\": " << samples[(samples.size() * 95 - 1) / 100]
                    << ", \"effective_payload_GB_per_s\": " << (bytes / average / 1000.0)
                    << ", \"correct\": true}";
            }
        }
        // A short, bounded copy load lets the launcher sample active PCIe width/gen.
        // This is not included in the latency/bandwidth measurements above.
        std::cerr << "PCIe sampling load: 3 seconds of 16 MiB staged copies.\n" << std::flush;
        const auto loadEnd = Clock::now() + std::chrono::seconds(3);
        while (Clock::now() < loadEnd) {
            stagedCopy(devices[0], devices[1], stage, kMaximumBytes);
            stagedCopy(devices[1], devices[0], stage, kMaximumBytes);
        }
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaStreamDestroy(devices[i].stream));
            CUDA(cudaFree(devices[i].buffer));
        }
        CUDA(cudaFreeHost(stage));
        out << "\n  ],\n  \"limitations\": ["
            << "\"Single run under the current Windows/display workload; WDDM scheduling can add jitter.\", "
            << "\"Serial pinned-host staging baseline, not a pipelined transfer implementation or collective benchmark.\", "
            << "\"Effective bandwidth counts payload once; each staged transfer crosses PCIe twice.\", "
            << "\"Eight warmup iterations do not establish sustained clocks; the three-second load runs after all measured tests.\", "
            << "\"No inference, tensor-core, GEMM, attention, or quantization throughput is measured.\", "
            << "\"Only CUDA ordinals 0 and 1 are transfer-benchmarked; all visible devices are inventoried.\"]\n}\n";
        std::cout << out.str();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gpu_probe: " << e.what() << '\n';
        return 1;
    }
}
