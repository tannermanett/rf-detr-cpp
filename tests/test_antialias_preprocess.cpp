#include "internal/cuda_preprocess.cuh"
#include "internal/cuda_check.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    cv::Mat image(4, 6, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                image.at<cv::Vec3b>(y, x)[c] = (y * 71 + x * 29 + c * 43) % 256;
            }
        }
    }
    // Frozen torchvision 0.20.1 / PyTorch 2.5.1 CUDA float32 reference:
    // BGR->RGB, /255, resize([3,3], antialias=True), ImageNet normalize.
    const std::array<float, 27> expected{
        0.074309289f, 0.796729267f, 0.720646441f,
        0.280784726f, -0.045808643f, -0.763825178f,
        -0.827920735f, -0.504751921f, 0.382065505f,
        -0.547368884f, 0.359243691f, 0.689626098f,
        0.944227874f, 0.730392516f, -0.443827361f,
        -0.701430917f, -0.467086852f, -0.232742831f,
        -1.072169185f, -0.169585943f, 0.732997417f,
        0.412798077f, 1.036514282f, 0.305733025f,
        1.451578259f, -0.099869303f, -0.567718506f};
    rfdetr::ImagePreprocessor preprocessor(3);
    preprocessor.set_antialias(true);
    auto device = rfdetr::dev_alloc(expected.size() * sizeof(float), "antialias test");
    preprocessor.process(image, static_cast<float*>(device.get()), nullptr);
    std::array<float, 27> actual{};
    RFDETR_CUDA_CHECK(cudaMemcpy(actual.data(), device.get(), sizeof(actual), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 3e-6f) {
            std::cerr << "antialias mismatch at " << i << ": " << actual[i]
                      << " vs " << expected[i] << '\n';
            return 1;
        }
    }
    // Edge normalization must retain a single source pixel when magnifying.
    image = cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 127, 255));
    preprocessor.process(image, static_cast<float*>(device.get()), nullptr);
    RFDETR_CUDA_CHECK(cudaMemcpy(actual.data(), device.get(), sizeof(actual), cudaMemcpyDeviceToHost));
    const float channels[] = {(1.0f - .485f) / .229f,
                              (127.0f / 255.0f - .456f) / .224f, -.406f / .225f};
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - channels[i / 9]) > 3e-6f) return 1;
    }
    return 0;
}
