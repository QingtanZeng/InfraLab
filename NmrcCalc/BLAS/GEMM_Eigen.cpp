#include <vector>
#include <chrono>
#include <concepts>
#include <mdspan>
#include <cstdlib>
#include <new>

#include <iostream>
#include <format>

#include <immintrin.h>

#include <Eigen/Core>

int main(){
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    constexpr size_t K = 1024;

    // memory allocation
    std::vector<float, AlignedAllocator<float, 32>> Araw(M * K, 1.0f);
    std::vector<float, AlignedAllocator<float, 32>> Braw(K * N, 2.0f);
    std::vector<float, AlignedAllocator<float, 32>> Craw(M * N, 0.0f);

    Eigen::Matrix<float, M, N> A


    auto start = std::chrono::high_resolution_clock::now();

    sgemm_avx2_col(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double gflops = (2.0 * M * N * K) / (duration.count() * 1e6);

    std::cout << std::format("Layout: Column-Major (std::layout_left)\n");
    std::cout << std::format("Time taken: {:.2f} ms\n", duration.count());
    std::cout << std::format("Performance: {:.2f} GFLOPS\n", gflops);

    return 0;
}