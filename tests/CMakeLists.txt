cmake_minimum_required(VERSION 3.20)

# Also configurable directly on hosts without CUDA, TensorRT, or OpenCV.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(rfdetr_budget_tests LANGUAGES CXX)
    enable_testing()
endif()

find_package(Threads REQUIRED)
set(RFDETR_TEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(test_gpu_budget test_gpu_budget.cpp
    "${RFDETR_TEST_ROOT}/src/core/gpu_budget.cpp")
target_include_directories(test_gpu_budget PRIVATE "${RFDETR_TEST_ROOT}/src")
target_compile_features(test_gpu_budget PRIVATE cxx_std_17)
target_link_libraries(test_gpu_budget PRIVATE Threads::Threads)
add_test(NAME gpu_budget COMMAND test_gpu_budget)

# Exercise real allocation wrappers with failure-injectable CUDA calls.
add_executable(test_cuda_budget test_cuda_budget.cpp
    "${RFDETR_TEST_ROOT}/src/core/gpu_budget.cpp"
    "${RFDETR_TEST_ROOT}/src/core/log.cpp")
target_include_directories(test_cuda_budget PRIVATE stubs
    "${RFDETR_TEST_ROOT}/src" "${RFDETR_TEST_ROOT}/include")
target_compile_features(test_cuda_budget PRIVATE cxx_std_17)
target_link_libraries(test_cuda_budget PRIVATE Threads::Threads)
add_test(NAME cuda_budget COMMAND test_cuda_budget)
