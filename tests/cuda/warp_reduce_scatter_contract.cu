// Standalone contract check, independent of PyTorch and the rasterizer VJP.
// nvcc -std=c++17 -Igsplat/cuda/include \
//   tests/cuda/warp_reduce_scatter_contract.cu -o /tmp/scatter-contract
#include "warp_reduce_scatter.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(operation)                                                  \
    do {                                                                       \
        cudaError_t error = (operation);                                        \
        if (error != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s: %s\n", #operation,                       \
                         cudaGetErrorString(error));                           \
            return false;                                                      \
        }                                                                      \
    } while (false)

template <int DIM>
__global__ void check_scatter(float *sums, int *owners, int *errors) {
    const auto block = cooperative_groups::this_thread_block();
    const cooperative_groups::thread_block_tile<32> warp =
        cooperative_groups::tiled_partition<32>(block);
    const int lane = warp.thread_rank();
    const int group = blockIdx.x * (block.size() / 32) + block.thread_rank() / 32;
    float contributions[DIM];
    for (int k = 0; k < DIM; ++k) {
        contributions[k] = float((lane * 17 + k * 13 + group) % 37 - 18);
    }
    const auto result = gsplat::warp_reduce_scatter(contributions, warp);
    if (result.owner) {
        if (result.component < 0 || result.component >= DIM) {
            atomicAdd(errors, 1);
        } else {
            sums[group * DIM + result.component] = result.value;
            atomicAdd(owners + group * DIM + result.component, 1);
        }
    }
}

template <int DIM> bool run_shape(dim3 threads) {
    constexpr int blocks = 3;
    const int warps_per_block = threads.x * threads.y * threads.z / 32;
    const int groups = blocks * warps_per_block;
    float *sums;
    int *owners, *errors;
    CUDA_CHECK(cudaMalloc(&sums, groups * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&owners, groups * DIM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&errors, sizeof(int)));
    CUDA_CHECK(cudaMemset(sums, 0, groups * DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(owners, 0, groups * DIM * sizeof(int)));
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(int)));
    check_scatter<DIM><<<blocks, threads>>>(sums, owners, errors);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> host_sums(groups * DIM);
    std::vector<int> host_owners(groups * DIM);
    int host_errors;
    CUDA_CHECK(cudaMemcpy(host_sums.data(), sums, host_sums.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_owners.data(), owners, host_owners.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(sums));
    CUDA_CHECK(cudaFree(owners));
    CUDA_CHECK(cudaFree(errors));
    if (host_errors != 0) return false;
    for (int group = 0; group < groups; ++group) {
        for (int k = 0; k < DIM; ++k) {
            float expected = 0;
            for (int lane = 0; lane < 32; ++lane) {
                expected += float((lane * 17 + k * 13 + group) % 37 - 18);
            }
            const int slot = group * DIM + k;
            if (host_owners[slot] != 1 || host_sums[slot] != expected) {
                std::fprintf(stderr, "DIM=%d group=%d component=%d owners=%d sum=%g expected=%g\n",
                             DIM, group, k, host_owners[slot], host_sums[slot], expected);
                return false;
            }
        }
    }
    return true;
}

template <int DIM> bool run_dimensions() {
    if (!run_shape<DIM>(dim3(8, 4)) || !run_shape<DIM>(dim3(8, 4, 2))) return false;
    std::printf("DIM=%d: unique owners and exact sums passed\n", DIM);
    if constexpr (DIM > 1) return run_dimensions<DIM - 1>();
    return true;
}

#ifdef GSPLAT_TEST_INVALID_DIM
// Compile with -DGSPLAT_TEST_INVALID_DIM to verify the public dimension guard.
template __global__ void check_scatter<17>(float *, int *, int *);
#endif

int main() {
    return run_dimensions<16>() ? 0 : 1;
}
