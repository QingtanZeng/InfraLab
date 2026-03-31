#include <vector>
#include <chrono>
#include <concepts>
#include <mdspan>
#include <cstdlib>

#include <iostream>
#include <format>

#include <immintrin.h>

template <typenmae T>
concept F32 = std::same_as<T, float>;

template <F32 T>
using MatrixViewCol = 
    std:mdspan<T, std::dextents<size_t, 2>, std::layout_left>;

// 1. 定义一个通用的对齐分配器
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        // C++17 标准的对齐内存分配
        void* ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        std::free(ptr);
    }
};

/**
 * 针对列优先 (Column-Major) 优化的单精度 GEMM (AVX2 + FMA)
 * 计算: C = A * B + C
 */
template <F32 T>
void sgemm_avx2_col(MatrixViewCol<T> A, MatrixViewCol<T> B, MatrixViewCol<T> C){
    const size_t M = A.extent(0);
    

}


 int main(){
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;
    constexpr size_t K = 1024;

    // memory allocation
    std::vector<float, AlignedAllocator<float, 32>> Araw(M*K, 1.0f); 
    std::vector<float, AlignedAllocator<float, 32>> Braw(K*N, 2.0f); 
    std::vector<float, AlignedAllocator<float, 32>> Craw(M*N, 0.0f); 

    // column-major view of raw matrix data
    MatrixViewCol<float> A(Araw.data(), M, K);
    MatrixViewCol<float> B(Braw.data(), K, N);
    MatrixViewCol<float> C(Craw.data(), M, N);

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