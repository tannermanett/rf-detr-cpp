#pragma once

// Test double only; production targets must use the CUDA toolkit header.
#include <cstddef>
#include <cstdlib>
#include <unordered_set>

using cudaError_t = int;
inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaError_t cudaErrorMemoryAllocation = 2;
inline constexpr cudaError_t cudaErrorUnknown = 999;

namespace fake_cuda {
inline bool fail_device = false;
inline bool fail_host = false;
inline bool fail_free = false;
inline int malloc_calls = 0;
inline std::unordered_set<void*> device;
inline std::unordered_set<void*> host;
}

inline const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }

inline cudaError_t cudaMalloc(void** p, std::size_t bytes) {
    ++fake_cuda::malloc_calls;
    if (fake_cuda::fail_device) return cudaErrorMemoryAllocation;
    *p = std::malloc(bytes);
    if (!*p) return cudaErrorMemoryAllocation;
    fake_cuda::device.insert(*p);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* p) {
    if (fake_cuda::fail_free) return cudaErrorUnknown;
    fake_cuda::device.erase(p);
    std::free(p);
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void** p, std::size_t bytes) {
    if (fake_cuda::fail_host) return cudaErrorMemoryAllocation;
    *p = std::malloc(bytes);
    if (!*p) return cudaErrorMemoryAllocation;
    fake_cuda::host.insert(*p);
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void* p) {
    fake_cuda::host.erase(p);
    std::free(p);
    return cudaSuccess;
}
