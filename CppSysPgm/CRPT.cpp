#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <tuple>
#include <optional>
#include <type_traits>

#include <cmath>
#include <numeric>
#include <Eigen/Dense>

#include <cassert>

template<typename Derived>
class LinearSolverBase {
public:
    struct SolverStatus{
        bool status;
        std::optional<Eigen::VectorXd> solution;
    } solverstatus;
    std::remove_const_t<decltype(std::declval<Eigen::MatrixXd>().ldlt())> ldltcache;

    const SolverStatus& solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b){
        return static_cast<Derived*>(this)->solverImpl(A,b);
    }
};

class LDLTSolver : public LinearSolverBase<LDLTSolver> {
public:
    auto solverImpl(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) ->const SolverStatus& {
       ldltcache.compute(A);
       
       if(ldltcache.info() != Eigen::Success || ! ldltcache.isPositive()){
            solverstatus.status = false;
            solverstatus.solution = std::nullopt;

            return solverstatus;
       }

        solverstatus.status = true;
        solverstatus.solution = ldltcache.solve(b);

        return solverstatus;
    }

};

void processSolver(std::unique_ptr<LDLTSolver> solver_ptr){
    std::cout << "Solver ownership transferred. Address: " << solver_ptr.get() << std::endl;
}

int main() {
    // 构造测试数据
    int n = 3;
    Eigen::MatrixXd A(n, n);
    A << 4, -1, 2, 
        -1, 6, 0, 
         2, 0, 5; // 正定矩阵
    Eigen::VectorXd b(n);
    b << 1, 2, 3;

    auto solver = std::make_unique<LDLTSolver>();

    auto [isvalid, result] = solver->solve(A,b);
    if (isvalid && result.has_value()) {
        std::cout << "Solution found:\n" << result.value() << std::endl;
    } else {
        std::cerr << "Decomposition failed!" << std::endl;
    }

    processSolver(std::move(solver));

    if (!solver) {
        std::cout << "Original pointer is now empty (as expected)." << std::endl;
    }

    return 0;

}