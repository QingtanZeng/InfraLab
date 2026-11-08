#include <cstddef>
#include <iostream>
#include <vector>
#include <span>
#include <mdspan>
#include <cmath>
#include <iomanip>
#include <expected>
#include <string>
#include <format>
#include <algorithm>
#include <cassert>

using VecView = std::span<double>;
using VecViewConst = std::span<const double>;
using Mat2DColView = std::mdspan<double, std::dextents<std::size_t, 2>, std::layout_left>;
using Mat2DColViewConst = std::mdspan<const double, std::dextents<std::size_t, 2>, std::layout_left>;

/* LDLT decomposition */
class LDLT {
public:
    size_t dim_;                       // Symmetric matrix dimension
    // Packed Lower Triangular Format
    std::vector<double> data_;      // n*(n+1)/2 elements for Lij and Dii (i >= j)

    explicit LDLT(size_t dim) : dim_(dim), data_(dim * (dim + 1) / 2, 0.0) {}

    // (idxR, idxC) compressed to column-major packed index for lower triangular elements (idxR >= idxC)
    [[nodiscard]] constexpr size_t index(size_t idxR, size_t idxC) const noexcept {
        assert((idxR >= idxC)&& "Rows must be more than columns!");
        return (idxC * dim_ - idxC * (idxC - 1) / 2 + (idxR - idxC));
    }

    // Dii
    [[nodiscard]] double D(size_t idx) const noexcept {
        return data_[index(idx, idx)];
    }

    // Lij: 自动处理对角元 1.0 与上三角 0.0
    [[nodiscard]] double L(size_t idxR, size_t idxC) const noexcept {
        if (idxR < idxC)
            return 0.0;
        else if (idxR == idxC)
            return 1.0;
        else
            return data_[index(idxR, idxC)];
    }

    /**
     * @brief 求解线性方程组 A * x = b (即 L * D * L^T * x = b)
     * @param b 右端向量，长度需为 dim_
     * @return 解向量 x 或错误描述
     * L(y) = b, D(z) = y, L^T*x = z
     */
    [[nodiscard]] std::expected<std::vector<double>, std::string>
    solver(VecViewConst b) const{
        assert((b.size() == dim_) && "Dimension is NOT equal !" );
        std::vector<double> x(dim_);    // b > y > z > x

        // Step1: L(y) = b
        double sum = 0.0;
        for(size_t i=0; i < dim_ ;++i){
            sum = 0.0;
            for(size_t k=0; k < i; ++k){
                sum += L(i, k) * x[k];
            }
            x[i] = b[i] - sum;    // Lii = 1.0
        }
        // Step2: D(z) = y
        for(size_t i=0; i < dim_; ++i){
            x[i] = x[i] / D(i);
        }
        // Step3: L^T*(x) = z
        for(size_t i=dim_; i > -1; --i ){
            sum = 0.0;
            for(size_t k=dim_; k > i; --k){
                sum += L(k, i) * x[k];
            }
            x[i] = x[i] - sum;    // Lii = 1.0
        }

        return x;
    }

    /**
     * @brief 重构原矩阵 A = L * D * L^T 用于数值残差检验
     */

};

/**
 * @brief 执行稠密对称矩阵的 LDLT 分解
 * @param n 矩阵维度
 * @param mat 列主序稠密对称矩阵 (长度 n*n)，mat[i, j]
 * @param tol 主元为 0 的容差
 * @return std::expected 包含紧凑分解结果或错误信息
 */
[[nodiscard]] std::expected<LDLT, std::string>
LDLTDecomp(Mat2DColViewConst mat, double tol = 1e-7) {
    size_t dim = mat.extent(0);
    if (dim != mat.extent(1)) {
        return std::unexpected(std::format("Matrix dimension mismatch: expected ({}x{}), got ({}x{})",
                                           dim, dim, mat.extent(0), mat.extent(1)));
    }
    LDLT ldlt(dim);
    std::vector<double> DkLjk(dim, 0.0);

    // Column-major calculation: index starting from 0
    for(size_t j = 0; j < dim; ++j){
        // Step01: precaculation DkLjk[k] = D[k] * L[j, k]
        for(size_t k = 0; k < j; ++k){               // when j=0, skip automatically.
            DkLjk[k] = ldlt.D(k) * ldlt.L(j,k);
        }

        // Step02: Dj = Aj - sum_{k=0}^{j-1} ((Dk*Ljk)*L[j, k])
        double sum = 0.0;
        for(size_t k=0; k < j; ++k){
            sum += DkLjk[k]*ldlt.L(j, k);
        }
        double dj = mat[j, j] - sum;
        if(std::abs(dj) < tol){
            return std::unexpected("Matrix is singular or requires pivoting at column " + std::to_string(j));;
        }
        ldlt.data_[ldlt.index(j,j)] = dj;

        // Step03: Lij = 1/Dj * (Aij - sum_{k=0}^{j-1} ((Dk*Ljk)*L[i, k]))
        for(size_t i = j+1; i < dim; ++i){
            sum = 0.0;
            for(size_t k=0; k < j; ++k){ 
                sum += DkLjk[k] * ldlt.L(i, k);
            }
            double Lij = (mat[i ,j] - sum) / dj;
            ldlt.data_[ldlt.index(i, j)] = Lij;
        }
    }

    return ldlt;
}

int main() {
    constexpr size_t n = 3;
    // 3x3 对称正定矩阵 (以列主序 std::layout_left 存储):
    // [  4  12 -16 ]
    // [ 12  37 -43 ]
    // [-16 -43  98 ]
    std::vector<double> A_data = {
        4.0,  12.0, -16.0,   // Column 0
        12.0,  37.0, -43.0,  // Column 1
        -16.0, -43.0,  98.0  // Column 2
    };

    Mat2DColViewConst A(A_data.data(), n, n);

    auto result = LDLTDecomp(n, A);
    if (!result) {
        std::cerr << "LDLT Decomposition failed: " << result.error() << "\n";
        return 1;
    }

    const auto& ldlt = result.value();
    std::cout << "===========================================\n";
    std::cout << "LDLT Decomposition Result (" << n << "x" << n << " Matrix)\n";
    std::cout << "===========================================\n\n";

    std::cout << "--- Diagonal Matrix D ---\n";
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  D[" << i << "] = " << ldlt.D(i) << "\n";
    }

    std::cout << "\n--- Unit Lower Triangular Matrix L ---\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << ldlt.L(i, j) << " ";
        }
        std::cout << "\n";
    }

    // 验证重构矩阵 ||A - L*D*L^T||
    auto A_rec = ldlt.reconstruct();
    Mat2DColViewConst A_rec_view(A_rec.data(), n, n);
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            max_err = std::max(max_err, std::abs(A[i, j] - A_rec_view[i, j]));
        }
    }
    std::cout << "\n--- Reconstruction Check ---\n";
    std::cout << "  Max error ||A - L*D*L^T||: " << std::scientific << max_err << "\n";

    // 测试线性方程组求解 A * x = b
    // 设定精确解 x_true = [1.0, 2.0, 3.0]^T
    // 计算 b = A * x_true:
    // b = [ 4*1 + 12*2 - 16*3, 12*1 + 37*2 - 43*3, -16*1 - 43*2 + 98*3 ]^T = [ -20, -43, 192 ]^T
    std::vector<double> b = {-20.0, -43.0, 192.0};
    auto x_solved = ldlt.solve(b);

    if (x_solved) {
        std::cout << "\n--- Linear System Solver (A * x = b) ---\n";
        std::cout << "  Target RHS b: [ -20.0, -43.0, 192.0 ]^T\n";
        std::cout << "  Computed  x: [ ";
        for (double val : x_solved.value()) {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "]^T (Expected: [ 1.0000 2.0000 3.0000 ]^T)\n";
    } else {
        std::cerr << "Solver failed: " << x_solved.error() << "\n";
    }

    return 0;
}