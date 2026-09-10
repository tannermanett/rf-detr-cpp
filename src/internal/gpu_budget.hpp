#pragma once

#include <cstddef>
#include <string>

namespace rfdetr::gpu_budget {

[[nodiscard]] std::string human(std::size_t bytes);

// Parse a binary byte size: "6G", "512M", "1024K", or a bare byte count.
// Empty and zero mean unlimited. Malformed values throw std::runtime_error.
[[nodiscard]] std::size_t parse_byte_size(const std::string& text);

// Process-local ceiling read once from RFDETR_GPU_MEM_LIMIT. Zero means unlimited.
// A malformed value remains a hard error on every call for the life of the process.
[[nodiscard]] std::size_t limit();

// Charge and release device allocations tracked by librfdetr. reserve() throws
// without changing the counter when the configured ceiling would be exceeded.
void reserve(std::size_t bytes, const char* what);
void release(std::size_t bytes) noexcept;

[[nodiscard]] std::size_t in_use() noexcept;
[[nodiscard]] std::size_t peak() noexcept;

// Test-only state controls. Production code must not call these.
void reset_for_testing(std::size_t limit_bytes) noexcept;
void reset_from_environment_for_testing() noexcept;

class Reservation {
public:
    Reservation() noexcept = default;
    Reservation(std::size_t bytes, const char* what) : bytes_(bytes) {
        reserve(bytes, what);
    }
    ~Reservation() { release(bytes_); }

    Reservation(Reservation&& other) noexcept : bytes_(other.bytes_) {
        other.bytes_ = 0;
    }

    Reservation& operator=(Reservation&& other) noexcept {
        if (this != &other) {
            release(bytes_);
            bytes_ = other.bytes_;
            other.bytes_ = 0;
        }
        return *this;
    }

    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

private:
    std::size_t bytes_{0};
};

}  // namespace rfdetr::gpu_budget
