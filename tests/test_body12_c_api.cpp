#undef NDEBUG
#include <cassert>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "rfdetr/c_api.h"
#include "internal/trt_session.hpp"

int main(int argc, char** argv) {
    assert(rfdetr_body12_create(nullptr) == nullptr);
    assert(std::strstr(rfdetr_last_error(), "path") != nullptr);
    assert(rfdetr_body12_infer(nullptr, nullptr, 0, nullptr, 0, nullptr, 0,
                             nullptr, 0) == 0);
    assert(std::strstr(rfdetr_last_error(), "buffers") != nullptr);
    rfdetr_body12_destroy(nullptr);
    assert(rfdetr_body12_cuda_graph_active(nullptr) == 0);
    assert(rfdetr_body12_create_ex(nullptr, 0) == nullptr);
    if (argc < 2) return 0;  // GPU/engine integration check is explicit.

    rfdetr::TrtSession reference(argv[1]);
    using Handle = std::unique_ptr<rfdetr_body12_t, decltype(&rfdetr_body12_destroy)>;
    Handle plain(rfdetr_body12_create_ex(argv[1], 0), rfdetr_body12_destroy);
    Handle graph(rfdetr_body12_create(argv[1]), rfdetr_body12_destroy);
    assert(plain && graph);
    std::vector<float> input(3 * 1024 * 1024);
    std::vector<float> expected[3] = {std::vector<float>(1200), std::vector<float>(600),
                                     std::vector<float>(28800)};
    std::vector<float> actual[3] = {expected[0], expected[1], expected[2]};
    const char* names[] = {"dets", "labels", "keypoints"};
    for (int frame = 0; frame < 6; ++frame) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>((i + frame * 17) % 251) / 125.f - 1.f;
        }
        reference.set_input("input", input.data(), input.size() * sizeof(float));
        reference.infer();
        for (int i = 0; i < 3; ++i) {
            reference.get_output_f32(names[i], expected[i].data(), expected[i].size());
        }
        for (auto* handle : {plain.get(), graph.get()}) {
            for (auto& output : actual) std::fill(output.begin(), output.end(), -999.f);
            // Invalid buffers must not enqueue work or overwrite caller outputs.
            assert(rfdetr_body12_infer(handle, input.data(), input.size() - 1,
                actual[0].data(), actual[0].size(), actual[1].data(), actual[1].size(),
                actual[2].data(), actual[2].size()) == 0);
            for (const auto& output : actual) {
                assert(std::all_of(output.begin(), output.end(), [](float v) { return v == -999.f; }));
            }
            assert(rfdetr_body12_infer(handle, input.data(), input.size(),
                actual[0].data(), actual[0].size(), actual[1].data(), actual[1].size(),
                actual[2].data(), actual[2].size()) == 1);
            for (int i = 0; i < 3; ++i) assert(actual[i] == expected[i]);
        }
    }
    assert(rfdetr_body12_cuda_graph_active(plain.get()) == 0);
    const bool active = rfdetr_body12_cuda_graph_active(graph.get()) != 0;
    std::cout << "Body12 exact output parity passed; graph_active=" << active << '\n';
    if (argc > 2 && std::strcmp(argv[2], "--require-graph") == 0) assert(active);
    if (argc > 2 && std::strcmp(argv[2], "--require-fallback") == 0) assert(!active);
}
