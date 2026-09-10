// Self-check for the GPU memory budget. Pure CPU — no CUDA, no GPU required,
// so this is the one part of the memory-limit work that can be verified off-box.
#undef NDEBUG          // asserts must survive a Release build
#include <cassert>

#include "internal/gpu_budget.hpp"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace bud = rfdetr::gpu_budget;

#define EXPECT_THROW(expr)                                                    \
    do {                                                                      \
        bool threw = false;                                                   \
        try { (void)(expr); } catch (const std::exception&) { threw = true; } \
        assert(threw && "expected a throw: " #expr);                          \
    } while (0)

static void test_parse() {
    assert(bud::parse_byte_size("")       == 0);
    assert(bud::parse_byte_size("   ")    == 0);
    assert(bud::parse_byte_size("0")      == 0);
    assert(bud::parse_byte_size("4096")   == 4096);
    assert(bud::parse_byte_size(" 4096 ") == 4096);

    assert(bud::parse_byte_size("1K")   == (1ull << 10));
    assert(bud::parse_byte_size("512M") == (512ull << 20));
    assert(bud::parse_byte_size("6G")   == (6ull << 30));

    // Suffix spellings must agree — a boot script writing "6GB" must not
    // silently mean something different from "6G".
    assert(bud::parse_byte_size("6g")   == bud::parse_byte_size("6G"));
    assert(bud::parse_byte_size("6GB")  == bud::parse_byte_size("6G"));
    assert(bud::parse_byte_size("6GiB") == bud::parse_byte_size("6G"));
    assert(bud::parse_byte_size("6gib") == bud::parse_byte_size("6G"));
    assert(bud::parse_byte_size("100B") == 100);

    EXPECT_THROW(bud::parse_byte_size("abc"));
    EXPECT_THROW(bud::parse_byte_size("6X"));
    EXPECT_THROW(bud::parse_byte_size("-1"));
    EXPECT_THROW(bud::parse_byte_size("6 G"));
    EXPECT_THROW(bud::parse_byte_size("G6"));
    EXPECT_THROW(bud::parse_byte_size("1iB"));
    EXPECT_THROW(bud::parse_byte_size("1.5G"));
    EXPECT_THROW(bud::parse_byte_size("+1G"));
    EXPECT_THROW(bud::parse_byte_size("99999999999999999999999"));
    EXPECT_THROW(bud::parse_byte_size("17179869184G"));   // 2^34 GiB overflows
}

static void test_reserve_release() {
    bud::reset_for_testing(1000);
    assert(bud::in_use() == 0);
    assert(bud::peak()   == 0);

    bud::reserve(400, "a");
    assert(bud::in_use() == 400);
    assert(bud::peak()   == 400);

    bud::reserve(600, "b");            // exactly fills the budget
    assert(bud::in_use() == 1000);
    assert(bud::peak()   == 1000);

    // Over the ceiling: must throw AND leave the counter untouched, otherwise a
    // failed allocation would permanently shrink the budget.
    EXPECT_THROW(bud::reserve(1, "c"));
    assert(bud::in_use() == 1000);

    bud::release(600);
    assert(bud::in_use() == 400);
    assert(bud::peak()   == 1000);     // peak is a high-water mark, it never drops

    bud::reserve(600, "d");            // room again
    assert(bud::in_use() == 1000);

    bud::reserve(0, "zero");           // no-op, must not throw even when full
    assert(bud::in_use() == 1000);
}

static void test_unlimited() {
    bud::reset_for_testing(0);         // 0 == unlimited
    bud::reserve(1ull << 40, "huge");
    assert(bud::in_use() == (1ull << 40));
    assert(bud::peak()   == (1ull << 40));   // still accounted, just never rejected
    bud::release(1ull << 40);
    assert(bud::in_use() == 0);
}

static void test_release_underflow_is_clamped() {
    bud::reset_for_testing(1000);
    bud::reserve(100, "a");
    bud::release(999999);              // buggy caller
    assert(bud::in_use() == 0);        // must clamp, not wrap to ~2^64
    bud::reserve(500, "b");            // and the budget must still work afterwards
    assert(bud::in_use() == 500);
}

// The error message is the whole point of the feature — it has to name the
// culprit, not a hardcoded stage.
static void test_error_message_names_the_caller() {
    bud::reset_for_testing(100);
    bool threw = false;
    try {
        bud::reserve(200, "mask decode scratch");
    } catch (const std::exception& e) {
        threw = true;
        const std::string msg = e.what();
        assert(msg.find("mask decode scratch") != std::string::npos);
        assert(msg.find("RFDETR_GPU_MEM_LIMIT") != std::string::npos);
    }
    assert(threw);
    assert(bud::in_use() == 0);
}

static void set_limit_env(const char* value) {
#ifdef _WIN32
    assert(_putenv_s("RFDETR_GPU_MEM_LIMIT", value ? value : "") == 0);
#else
    assert((value ? setenv("RFDETR_GPU_MEM_LIMIT", value, 1)
                  : unsetenv("RFDETR_GPU_MEM_LIMIT")) == 0);
#endif
}

static void test_environment() {
    set_limit_env("typo");
    bud::reset_from_environment_for_testing();
    EXPECT_THROW(bud::limit());
    EXPECT_THROW(bud::reserve(1, "retry"));
    EXPECT_THROW(bud::limit());
    assert(bud::in_use() == 0);

    set_limit_env("2MiB");
    bud::reset_from_environment_for_testing();
    assert(bud::limit() == (2ull << 20));
    set_limit_env("3G");
    assert(bud::limit() == (2ull << 20));  // read once per process

    const char* unlimited_values[] = {nullptr, "", "0"};
    for (const char* value : unlimited_values) {
        set_limit_env(value);
        bud::reset_from_environment_for_testing();
        assert(bud::limit() == 0);
    }
}

static void test_reservation_lifetime() {
    bud::reset_for_testing(100);
    {
        bud::Reservation first(60, "first");
        bud::Reservation moved(std::move(first));
        bud::Reservation second(40, "second");
        second = std::move(moved);
        assert(bud::in_use() == 60);
        EXPECT_THROW(bud::Reservation(41, "failed construction"));
        assert(bud::in_use() == 60);
        try {
            bud::Reservation temporary(40, "unwind");
            throw std::runtime_error("injected failure");
        } catch (const std::runtime_error&) {}
        assert(bud::in_use() == 60);
    }
    assert(bud::in_use() == 0);
    assert(bud::peak() == 100);
}

static void test_unlimited_counter_overflow() {
    bud::reset_for_testing(0);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    bud::reserve(maximum, "maximum");
    EXPECT_THROW(bud::reserve(1, "overflow"));
    assert(bud::in_use() == maximum);
    bud::release(maximum);
    assert(bud::in_use() == 0);
}

static void test_concurrent_ceiling() {
    bud::reset_for_testing(4);
    std::atomic<int> completed{0};
    std::atomic<int> accepted{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            bool held = false;
            try {
                bud::reserve(1, "concurrent");
                held = true;
                ++accepted;
            } catch (const std::runtime_error&) {}
            ++completed;
            while (completed.load() != 16) std::this_thread::yield();
            if (held) bud::release(1);
        });
    }
    for (auto& thread : threads) thread.join();
    assert(accepted == 4);
    assert(bud::peak() == 4);
    assert(bud::in_use() == 0);
}

int main() {
    test_parse();
    test_reserve_release();
    test_unlimited();
    test_release_underflow_is_clamped();
    test_error_message_names_the_caller();
    test_environment();
    test_reservation_lifetime();
    test_unlimited_counter_overflow();
    test_concurrent_ceiling();
    std::puts("gpu_budget: all checks passed");
    return 0;
}
