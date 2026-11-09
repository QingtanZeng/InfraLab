#include <__expected/expected.h>
#include <__expected/unexpected.h>
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

constexpr double epsilon = 1e-4;   // threshold of pivot elements
constexpr double delta = 1e-2;      // dynamic regularization



/* LDLT decomposition */
class LDLT {
public:
    size_t dim_;                    // Symmetric matrix dimension
    // Packed Lower Triangular Format
    std::vector<double> data_;      // n*(n+1)/2 elements for Lij and Dii (i >= j)
    size_t cntPivotsRglrzd = 0;     // regularized pivots counter

    struct SolveResult{
        std::vector<double> x;
        size_t iterIR = 0;
        double nrmRsdl = 0.0;       // final residual norm
        bool b_Convg = false;

        SolveResult() = default;
        explicit SolveResult(size_t dim) : x(dim, 0.0) {}
    };
    SolveResult slvres;

    explicit LDLT(size_t dim) : dim_(dim), data_(dim * (dim + 1) / 2, 0.0), slvres(dim) {}

    // (idxR, idxC) compressed to column-major packed index for lower triangular elements (idxR >= idxC)
    [[nodiscard]] constexpr size_t index(size_t idxR, size_t idxC) const noexcept {
        assert((idxR >= idxC) && "Row index must be greater than or equal to column index!");
        // 规避 idxC == 0 时无符号数 idxC - 1 的潜在下溢
        const size_t col_offset = (idxC == 0) ? 0 : (idxC * dim_ - idxC * (idxC - 1) / 2);
        return col_offset + (idxR - idxC);
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
    [[nodiscard]] std::expected<void, std::string>
    solve(VecViewConst b, VecView x) const {
        if (b.size() != dim_) {
            return std::unexpected(std::format("Dimension mismatch: expected {}, got {}", dim_, b.size()));
        }

        // 复用内存空间: b -> y -> z -> x

        // Step 1: 前向替换求解 L * y = b (L 为单位下三角矩阵, L_ii = 1.0)
        for (size_t i = 0; i < dim_; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) {
                sum += L(i, k) * x[k];
            }
            x[i] = b[i] - sum;    // Lii = 1.0
        }

        // Step 2: 对角缩放求解 D * z = y
        for (size_t i = 0; i < dim_; ++i) {
            x[i] = x[i] / D(i);
        }

        // Step 3: 后向替换求解 L^T * x = z (L^T 为单位上三角矩阵, (L^T)_ik = L_ki)
        // 注意: size_t 下溢安全的递减循环写法 (i-- > 0 从 dim_-1 遍历到 0)
        for (size_t i = dim_; i-- > 0; ) {
            double sum = 0.0;
            for (size_t k = i + 1; k < dim_; ++k) {
                sum += L(k, i) * x[k];
            }
            x[i] = x[i] - sum;    // Lii = 1.0
        }
        return {};
    }

    /* solve with dynamic regularization and iterative refinement */
    [[nodiscard]] std::expected<void, std::string>
    solve_IR(VecViewConst b, Mat2DColViewConst mat){
        std::vector<double> rsdl(dim_, 0.0);
        std::vector<double> dlt_x(dim_, 0.0);

        // Initial solve
        auto result = solve(b, slvres.x);
        if(!result){
            return std::unexpected(result.error());
        }

        // Iterative refinement, maximum iterations 3
        for(size_t iter = 0; iter < 10; ++iter){
            // calculate residual at k
            double nrmRsdlMax = 0.0;
            for(size_t idxR = 0; idxR < dim_; ++idxR){
                double ax_i = 0.0;
                for(size_t idxC = 0; idxC < dim_; ++idxC){
                    ax_i += mat[idxR, idxC] * slvres.x[idxC];
                }
                rsdl[idxR] = b[idxR] - ax_i;
                nrmRsdlMax = std::max(nrmRsdlMax, std::abs(rsdl[idxR]));
            }
            slvres.nrmRsdl = nrmRsdlMax;
            slvres.iterIR = iter + 1;

            // check terminal conditions
            if(nrmRsdlMax < 1e-7){
                slvres.b_Convg = true;
                break;
            }

            // do next refinement
            auto result = solve(rsdl, dlt_x);
            if (!result) {
                return std::unexpected(result.error());
            }
            for(size_t i = 0; i < dim_ ;++i){
                slvres.x[i] += dlt_x[i];
            }
        }

        return {};
    }

    /**
     * @brief 重构原矩阵 A = L * D * L^T 用于数值残差检验
     */
    [[nodiscard]] std::vector<double> reconstruct() const {
        std::vector<double> A_rec(dim_ * dim_, 0.0);
        Mat2DColView view(A_rec.data(), dim_, dim_);
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                double val = 0.0;
                size_t k_max = std::min(i, j);
                for (size_t k = 0; k <= k_max; ++k) {
                    val += L(i, k) * D(k) * L(j, k);
                }
                view[i, j] = val;
            }
        }
        return A_rec;
    }
};

/**
 * @brief 执行稠密对称矩阵的 LDLT 分解 with dynamic regularization
 * @param mat 列主序稠密对称矩阵 (长度 n*n)，mat[i, j]
 * @param tol 主元为 0 的容差
 * @return std::expected 包含紧凑分解结果或错误信息
 */
[[nodiscard]] std::expected<LDLT, std::string>
LDLTDecomp_Rglrzd(Mat2DColViewConst mat, double tol = 1e-7) {
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
        for (size_t k = 0; k < j; ++k) {
            sum += DkLjk[k] * ldlt.L(j, k);
        }
        double dj0 = mat[j, j] - sum;
        // Dynamic Regularization
        double sj = (dj0 >= double(0.0))? +1.0 : -1.0;   // retain sign of pivot element
        double dj = 0.0;
        if ( (sj*dj0) < epsilon) {
            dj = sj*delta;
            ldlt.cntPivotsRglrzd ++ ;
        }else{
            dj = dj0;
        }
        ldlt.data_[ldlt.index(j, j)] = dj;

        // Step03: Lij = 1/Dj * (Aij - sum_{k=0}^{j-1} ((Dk*Ljk)*L[i, k]))
        for(size_t i = j+1; i < dim; ++i){
            sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += DkLjk[k] * ldlt.L(i, k);
            }
            double Lij = (mat[i, j] - sum) / dj;
            ldlt.data_[ldlt.index(i, j)] = Lij;
        }
    }

    return ldlt;
}

int main() {
    constexpr size_t n = 3;
    // 构造一个经典的不定矩阵 (Indefinite Matrix，鞍点结构)：
    // [  0,  2,  1 ]
    // [  2,  0,  1 ]
    // [  1,  1, -2 ]
    // 首对角元 A(0,0) = 0，普通 LDLT 分解在第一步即直接发生除零崩溃！
    const std::vector<double> A_data = {
        0.0, 2.0,  1.0,   // 第 0 列
        2.0, 0.0,  1.0,   // 第 1 列
        1.0, 1.0, -2.0    // 第 2 列
    };

    Mat2DColViewConst A(A_data.data(), n, n);

    auto result = LDLTDecomp_Rglrzd(A);
    if (!result) {
        std::cerr << "LDLT Decomposition failed: " << result.error() << "\n";
        return 1;
    }
    auto& ldlt = result.value();
    std::cout << "分解成功! 共触发动态正则化主元数: " << ldlt.cntPivotsRglrzd << "\n\n";
    std::cout << "===========================================\n";
    std::cout << "LDLT Decomposition Result (" << n << "x" << n << " Matrix)\n";
    std::cout << "===========================================\n\n";
    std::cout << "--- Diagonal Matrix D ---\n";
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  D[" << i << "] = " << std::fixed << std::setprecision(15) 
                    << ldlt.D(i) << "\n";
    }
    std::cout << "\n--- Unit Lower Triangular Matrix L ---\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << std::setw(18) << std::fixed << std::setprecision(8) << ldlt.L(i, j) << " ";
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

    // 构造右端项 b: 设真实解 x* = [1.0, 2.0, 3.0]^T
    // A * x* = [ 0*1 + 2*2 + 1*3 ] = [  7.0 ]
    //          [ 2*1 + 0*2 + 1*3 ] = [  5.0 ]
    //          [ 1*1 + 1*2 - 2*3 ] = [ -3.0 ]
    const std::vector<double> b = { 7.0, 5.0, -3.0 };

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\n --- 测试: 对称不定矩阵 A (含对角线零元) 的 LDLT 分解 ---\n";
    auto x_solved = ldlt.solve(b, ldlt.slvres.x);

    if (x_solved) {
        std::cout << "\n--- Linear System Solver (A * x = b) ---\n";
        std::cout << "  Target RHS b: [1.0, 2.0, 3.0]^T\n";
        std::cout << "  Computed  x: [ ";
        for (double val : ldlt.slvres.x) {
            std::cout << std::fixed << std::setprecision(15) << val << " ";
        }
        std::cout << "]^T (包含 O(delta) 的扰动误差)\n (Expected: [ 1.0000 2.0000 3.0000 ]^T)\n";
    } else {
        std::cerr << "Solver failed: " << x_solved.error() << "\n";
    }
    // 3. 执行 Iterative Refinement (IR)
    auto x_IR = ldlt.solve_IR(b, A);

    std::cout << "\n执行迭代求精 (Iterative Refinement) 结果:\n";
    std::cout << "  收敛状态: " << ( ldlt.slvres.b_Convg ? "已收敛" : "未收敛") << "\n";
    std::cout << "  使用迭代步数: " << ldlt.slvres.iterIR << "\n";
    std::cout << "  最终残差无穷范数 ||b - Ax||_inf: " << std::fixed << std::setprecision(15) 
                << ldlt.slvres.nrmRsdl << "\n";
    std::cout << "最终高精度解向量 (期望值: 1.0, 2.0, 3.0):\n";
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  x[" << i << "] = " << std::fixed << std::setprecision(15) << ldlt.slvres.x[i] << "\n";
    }

    return 0;
}