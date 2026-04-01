#include <vector>
#include <chrono>
#include <concepts>
#include <mdspan>
#include <cstdlib>
#include <ranges>
#include <algorithm>

#include <numeric>
#include <execution>

#include <iostream>
#include <format>

#include <immintrin.h>


template <typename T>
concept F32 = std::same_as<T, float>;

template <F32 T>
using MatrixViewCol = 
    std::mdspan<T, std::dextents<size_t, 2>, std::layout_left>;

// 1. 定义一个通用的对齐分配器
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        // C++17 标准的对齐内存分配
        // 警告：std::aligned_alloc 要求申请的字节数必须是对齐量(Alignment)的整数倍
        std::size_t size = n * sizeof(T);
        std::size_t aligned_size = (size + Alignment - 1) & ~(Alignment - 1);
        void* ptr = std::aligned_alloc(Alignment, aligned_size);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        std::free(ptr);
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

/**
 * 针对列优先 (Column-Major) 优化的单精度 GEMM (AVX2 + FMA)
 * 计算: C = A * B + C
 * 多核并行idxj
 */
template <F32 T>
void sgemm_avx2_col_par(MatrixViewCol<T> A, MatrixViewCol<T> B, MatrixViewCol<T> C){
    const size_t M = A.extent(0);
    const size_t K = A.extent(1);
    const size_t N = C.extent(1);


    // 1. 零内存分配：生成一个虚拟的 [0, N-1] 视图
    auto col_indices = std::views::iota((size_t)0, N);

    // column-major circular order: j(N) > k(K) > i(M)
    // inner loop: C(i, J) = A(i, K) .* b(K, J) 
    // 2. 将外层循环替换为 C++17 的并行算法
    std::for_each(std::execution::par, col_indices.begin(), col_indices.end(), 
        [&](size_t idxj) {
            
            for(size_t idxk = 0; idxk < K; ++idxk){
            // at inner loop, B(k, j) as float constant scalor is broadcasted to
            // 8 slots of 256bits AVX2 register
                __m256 b_vec = _mm256_set1_ps(B(idxk, idxj));

                size_t idxi = 0;
                for(;idxi + 7 < M; idxi+=8){
                    // load A(i, K)~A(i+8, K)
                    __m256 a_vec = _mm256_loadu_ps(&A(idxi, idxk));
                    // load C(i,J)~C(i+8, J)
                    __m256 c_vec = _mm256_loadu_ps(&C(idxi, idxj));

                    // FMA: C(i ~ i+8) += A(i ~ i+8) .* b(K, J) 
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                    // write back to memory
                    _mm256_storeu_ps(&C(idxi, idxj), c_vec);
                }

                // 5. 标量收尾：处理 M 不是 8 的倍数时的剩余行元素
                for (; idxi < M; ++idxi) {
                    C(idxi, idxj) += A(idxi, idxk) * B(idxk, idxj);
                }
            }  
        }
    );
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

    sgemm_avx2_col_par(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double gflops = (2.0 * M * N * K) / (duration.count() * 1e6);

    std::cout << std::format("Layout: Column-Major (std::layout_left)\n");
    std::cout << std::format("Time taken: {:.2f} ms\n", duration.count());
    std::cout << std::format("Performance: {:.2f} GFLOPS\n", gflops);

    return 0;
 }