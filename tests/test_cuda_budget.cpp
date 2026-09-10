#undef NDEBUG
#include <cassert>

#include "internal/cuda_raii.hpp"

#include <cstdio>
#include <utility>

namespace bud = rfdetr::gpu_budget;

template <typename F>
static void expect_failure(F action) {
    bool threw = false;
    try { action(); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}

static void test_rejection_and_cuda_failure() {
    bud::reset_for_testing(100);
    const int calls = fake_cuda::malloc_calls;
    expect_failure([] { (void)rfdetr::dev_alloc(101, "over budget"); });
    assert(fake_cuda::malloc_calls == calls);
    assert(bud::in_use() == 0);
    fake_cuda::fail_device = true;
    expect_failure([] { (void)rfdetr::dev_alloc(80, "CUDA OOM"); });
    fake_cuda::fail_device = false;
    assert(bud::in_use() == 0);
    assert(fake_cuda::device.empty());
    auto retry = rfdetr::dev_alloc(100, "retry");
    assert(bud::in_use() == 100);
}

static void test_move_and_regrow() {
    bud::reset_for_testing(100);
    {
        auto first = rfdetr::dev_alloc(60, "first");
        auto moved = std::move(first);
        auto second = rfdetr::dev_alloc(40, "second");
        second = std::move(moved);
        assert(bud::in_use() == 60);
        second.reset();
        second = rfdetr::dev_alloc(100, "regrow");
        assert(bud::in_use() == 100);
        assert(bud::peak() == 100);
    }
    assert(bud::in_use() == 0);
    assert(fake_cuda::device.empty());
}

static void test_host_failure_unwinds_device() {
    bud::reset_for_testing(100);
    fake_cuda::fail_host = true;
    expect_failure([] {
        auto device = rfdetr::dev_alloc(100, "device before host failure");
        auto host = rfdetr::host_alloc(100);
    });
    fake_cuda::fail_host = false;
    assert(bud::in_use() == 0);
    assert(fake_cuda::device.empty());
    auto host = rfdetr::host_alloc(1000);
    assert(bud::in_use() == 0);
}

static void test_failed_free_keeps_charge() {
    bud::reset_for_testing(100);
    auto device = rfdetr::dev_alloc(100, "failed free");
    void* raw = device.get();
    fake_cuda::fail_free = true;
    device.reset();
    fake_cuda::fail_free = false;
    assert(bud::in_use() == 100);
    expect_failure([] { (void)rfdetr::dev_alloc(1, "still charged"); });
    rfdetr::detail::DevFree{100}(raw);  // Recover the injected failure for test cleanup.
    assert(bud::in_use() == 0);
}

int main() {
    test_rejection_and_cuda_failure();
    test_move_and_regrow();
    test_host_failure_unwinds_device();
    test_failed_free_keeps_charge();
    assert(!rfdetr::dev_alloc(0, "zero"));
    assert(!rfdetr::host_alloc(0));
    assert(fake_cuda::device.empty());
    assert(fake_cuda::host.empty());
    std::puts("cuda_budget: all checks passed (CUDA test double)");
}
