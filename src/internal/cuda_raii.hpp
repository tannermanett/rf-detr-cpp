#pragma once

#include "cuda_check.hpp"
#include "gpu_budget.hpp"
#include "rfdetr/core/log.hpp"

#include <cuda_runtime_api.h>
#include <memory>

namespace rfdetr {
namespace detail {

struct DevFree {
    std::size_t bytes{0};
    void operator()(void* p) const noexcept {
        if (!p) return;
        try {
            RFDETR_CUDA_CHECK(cudaFree(p));
            gpu_budget::release(bytes);
        } catch (const std::exception& error) {
            // Keep the charge if CUDA could not confirm that memory was freed.
            log_message(LogSeverity::kError, error.what());
        }
    }
};

struct HostFree {
    void operator()(void* p) const noexcept {
        if (!p) return;
        try {
            RFDETR_CUDA_CHECK(cudaFreeHost(p));
        } catch (const std::exception& error) {
            log_message(LogSeverity::kError, error.what());
        }
    }
};

}  // namespace detail

// Owning device-memory pointer.  Drop-in for void* managed with cudaMalloc/cudaFree.
using DevPtr  = std::unique_ptr<void, detail::DevFree>;

// Owning pinned-host-memory pointer.  Managed with cudaMallocHost/cudaFreeHost.
using HostPtr = std::unique_ptr<void, detail::HostFree>;

// Reset an old buffer before regrowing it, so both allocations are not charged
// at once. The deleter carries the reservation through moves and exceptions.
[[nodiscard]] inline DevPtr dev_alloc(std::size_t bytes, const char* what) {
    if (bytes == 0) return DevPtr{};
    gpu_budget::reserve(bytes, what);
    void* p = nullptr;
    try {
        RFDETR_CUDA_CHECK(cudaMalloc(&p, bytes));
    } catch (...) {
        gpu_budget::release(bytes);
        throw;
    }
    return DevPtr{p, detail::DevFree{bytes}};
}

// Pinned host memory is system RAM and does not consume the device budget.
[[nodiscard]] inline HostPtr host_alloc(std::size_t bytes) {
    if (bytes == 0) return HostPtr{};
    void* p = nullptr;
    RFDETR_CUDA_CHECK(cudaMallocHost(&p, bytes));
    return HostPtr{p};
}

}  // namespace rfdetr
