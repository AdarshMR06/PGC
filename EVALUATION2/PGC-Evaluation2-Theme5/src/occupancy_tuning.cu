#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(e) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void saxpyKernel(const float* x, const float* y, float* out,
                            size_t n, float a, int sharedBytes) {
    extern __shared__ unsigned char scratch[];
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Controlled shared-memory pressure experiment. Each thread writes one
    // byte when shared memory is requested; the barrier ensures use.
    if (sharedBytes > 0) {
        const unsigned int s = threadIdx.x % sharedBytes;
        scratch[s] = static_cast<unsigned char>(threadIdx.x & 0xff);
        __syncthreads();
        if (threadIdx.x == 0) {
            volatile unsigned char sink = scratch[0];
            (void)sink;
        }
    }

    if (i < n) out[i] = a * x[i] + y[i];
}

static double runCase(const std::vector<float>& hx,
                      const std::vector<float>& hy,
                      std::vector<float>& hout,
                      int blockSize, int sharedBytes,
                      int repetitions, float a) {
    const size_t n = hx.size();
    const size_t bytes = n * sizeof(float);
    float *dx = nullptr, *dy = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = static_cast<int>((n + blockSize - 1) / blockSize);
    // Warm-up
    saxpyKernel<<<blocks, blockSize, sharedBytes>>>(dx, dy, dout, n, a, sharedBytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    times.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        saxpyKernel<<<blocks, blockSize, sharedBytes>>>(dx, dy, dout, n, a, sharedBytes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    // Report median to reduce sensitivity to occasional OS/display interruptions.
    double median = times[times.size()/2];

    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dout));
    return median;
}

int main(int argc, char** argv) {
    const size_t n = (argc > 1) ? static_cast<size_t>(std::stoull(argv[1])) : (1ULL << 24);
    const int repetitions = (argc > 2) ? std::max(5, std::stoi(argv[2])) : 15;
    const float a = 2.5f;
    const int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    const int sharedCases[] = {0, 4096, 16384};

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));

    std::cout << "GPU: " << p.name << "\n"
              << "Compute capability: " << p.major << "." << p.minor << "\n"
              << "SM count: " << p.multiProcessorCount << "\n"
              << "Global memory (MiB): " << p.totalGlobalMem / (1024.0*1024.0) << "\n"
              << "Warp size: " << p.warpSize << "\n"
              << "Max threads/block: " << p.maxThreadsPerBlock << "\n"
              << "Registers/SM: " << p.regsPerMultiprocessor << "\n"
              << "Shared memory/SM (bytes): " << p.sharedMemPerMultiprocessor << "\n"
              << "Shared memory/block (bytes): " << p.sharedMemPerBlock << "\n"
              << "Vector elements: " << n << "\n"
              << "Timing repetitions: " << repetitions << "\n";

    std::vector<float> x(n), y(n), out(n), reference(n);
    for (size_t i=0; i<n; ++i) {
        x[i] = static_cast<float>((i % 1000) * 0.001);
        y[i] = static_cast<float>((i % 700) * 0.002);
        reference[i] = a*x[i] + y[i];
    }

    std::ofstream gpu("results/gpu_properties.csv");
    gpu << "property,value\n"
        << "gpu_name," << p.name << "\n"
        << "compute_capability," << p.major << "." << p.minor << "\n"
        << "sm_count," << p.multiProcessorCount << "\n"
        << "warp_size," << p.warpSize << "\n"
        << "max_threads_per_block," << p.maxThreadsPerBlock << "\n"
        << "registers_per_sm," << p.regsPerMultiprocessor << "\n"
        << "shared_memory_per_sm_bytes," << p.sharedMemPerMultiprocessor << "\n"
        << "shared_memory_per_block_bytes," << p.sharedMemPerBlock << "\n"
        << "total_global_memory_bytes," << p.totalGlobalMem << "\n";
    gpu.close();

    std::ofstream blockCsv("results/block_size_results.csv");
    blockCsv << "block_size,blocks_launched,warps_per_block,median_time_ms,relative_speed,validation\n";
    std::ofstream occCsv("results/occupancy_results.csv");
    occCsv << "block_size,registers_per_thread,active_blocks_per_sm,active_warps_per_sm,theoretical_occupancy,limiting_resources\n";
    std::ofstream sharedCsv("results/shared_memory_results.csv");
    sharedCsv << "block_size,shared_memory_bytes,median_time_ms,active_blocks_per_sm,theoretical_occupancy,validation\n";

    double bestMs = 1e30;
    int bestBlock = 0;
    std::vector<double> baselineTimes;

    for (int b : blockSizes) {
        int minGrid = 0, suggestedBlock = 0, activeBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &suggestedBlock,
                    saxpyKernel, 0, b));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &activeBlocks, saxpyKernel, b, 0));
        const int blocks = static_cast<int>((n+b-1)/b);
        double ms = runCase(x,y,out,b,0,repetitions,a);
        bool valid = true;
        for (size_t i=0; i<n; ++i) {
            if (std::fabs(out[i]-reference[i]) > 1e-5f) { valid=false; break; }
        }
        double warps = b / static_cast<double>(p.warpSize);
        int activeWarps = activeBlocks * static_cast<int>(warps);
        double occupancy = (activeWarps / static_cast<double>(p.maxThreadsPerMultiProcessor/p.warpSize))*100.0;
        occupancy = std::min(100.0, occupancy);
        blockCsv << b << "," << blocks << "," << warps << "," << std::setprecision(8)
                 << ms << ",," << (valid ? "PASS":"FAIL") << "\n";
        occCsv << b << ",compiler_reported_separately," << activeBlocks << ","
               << activeWarps << "," << occupancy
               << ",API resource model (register/shared-memory/kernel limits)\n";
        baselineTimes.push_back(ms);
        if (ms < bestMs) { bestMs=ms; bestBlock=b; }
        std::cout << "Baseline block=" << b << " blocks=" << blocks
                  << " warps/block=" << warps << " median_ms=" << ms
                  << " validation=" << (valid?"PASS":"FAIL")
                  << " active_blocks/SM=" << activeBlocks
                  << " theoretical_occupancy_estimate=" << occupancy << "%\n";
    }
    blockCsv.close();
    occCsv.close();

    for (int b : blockSizes) {
        for (int smem : sharedCases) {
            int activeBlocks = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, saxpyKernel, b, smem));
            double ms = runCase(x,y,out,b,smem,repetitions,a);
            bool valid = true;
            for (size_t i=0; i<n; ++i) {
                if (std::fabs(out[i]-reference[i]) > 1e-5f) { valid=false; break; }
            }
            int warpsPerBlock = b / p.warpSize;
            int maxWarpsPerSM = p.maxThreadsPerMultiProcessor / p.warpSize;
            double occ = std::min(100.0, activeBlocks * warpsPerBlock * 100.0 / maxWarpsPerSM);
            sharedCsv << b << "," << smem << "," << std::setprecision(8) << ms
                      << "," << activeBlocks << "," << occ << ","
                      << (valid?"PASS":"FAIL") << "\n";
            std::cout << "Shared block=" << b << " dynamic_smem=" << smem
                      << " median_ms=" << ms << " active_blocks/SM=" << activeBlocks
                      << " occupancy_estimate=" << occ << "% validation="
                      << (valid?"PASS":"FAIL") << "\n";
        }
    }
    sharedCsv.close();

    std::cout << "\nBest baseline block size by median kernel time: "
              << bestBlock << " threads (" << bestMs << " ms)\n";
    std::cout << "CSV files written to results/. Register usage must be taken from nvcc ptxas output.\n";
    return 0;
}
