#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numbers> // C++20 引入
#include <memory> // 用于 std::unique_ptr
#include <filesystem>

#include <cppad/cg/cppadcg.hpp>
#include <cppad/cppad.hpp>
#include <cppad/cg/math.hpp>

using scalar_t = double; // 定义标量类型为 double
using ad_base_t = CppAD::cg::CG<double>; // 定义自动微分基类型
using ad_scalar_t = CppAD::AD<ad_base_t>; // 
using ad_vector_t = std::vector<ad_scalar_t>; //


class FuncAD{
    private:
        std::string funcADName_;
        // [新增] 成员变量：用于持有加载好的动态库对象，防止函数结束时被销毁
        std::unique_ptr<CppAD::cg::DynamicLib<scalar_t>> dynamicLib_;
    public:
        FuncAD(const std::string& name): funcADName_(name){
            // basic info
            size_t n = 2; // 输入维度
            size_t m = 2; // 输出维度

            // Calculation Flow graph of function
            auto systemFlowMapFunc = [&](const ad_vector_t&x, ad_vector_t& y){
                    y[0] = x[0]+x[0]*x[1];
                    y[1] = x[1]*sin(x[0]);
            };

            // Independent variables
            ad_vector_t xad(n);
            for(size_t i=0; i<n; i++) xad[i] = 1.0;
            CppAD::Independent(xad);

            ad_vector_t yad(m);

            // Compute function values
            systemFlowMapFunc(xad, yad);

            CppAD::ADFun<ad_base_t> funcAD(xad, yad);

            // 4. 代码生成 (Code Generation)
            std::cout << "[INFO] Generating C source code..." << std::endl;

            CppAD::cg::ModelCSourceGen<scalar_t> cgen(funcAD, funcADName_);
            cgen.setCreateJacobian(true); // 生成雅可比矩阵代码
            cgen.setCreateHessian(false); // 不生成 Hessian 矩阵代码
            cgen.setCreateForwardZero(true); // 生成零阶前向模式代码

            CppAD::cg::ModelLibraryCSourceGen<scalar_t> libcgen(cgen);
            CppAD::cg::DynamicModelLibraryProcessor<scalar_t> p(libcgen);
            // 编译选项配置
            CppAD::cg::GccCompiler<scalar_t> compiler;
            std::vector<std::string> compilerOptions = compiler.getCompileFlags();
            // 1. 获取当前工作目录的绝对路径 (去除相对路径的不确定性)
            std::string cwd = std::filesystem::current_path().string();
            std::cout << "[DEBUG] Current Working Directory: " << cwd << std::endl;

            // 2. 添加优化和标准标志
            compilerOptions.push_back("-O3");
            compilerOptions.push_back("-fPIC");      // 生成动态库必须

            // 3. 拼接绝对路径传递给 -I
            // 确保这些路径下确实存在 include 文件夹
            compilerOptions.push_back("-I" + cwd + "/lib/cppad/install/include");
            compilerOptions.push_back("-I" + cwd + "/lib/cppadcodegen/install/include");
            
            // 4. [关键] 添加 Eigen 路径 (系统路径通常是固定的)
            compilerOptions.push_back("-I/usr/include/eigen3");

            // 打印一下参数供调试
            std::cout << "[DEBUG] Internal Compiler Flags: ";
            for(const auto& opt : compilerOptions) std::cout << opt << " ";
            std::cout << std::endl;
            // ------------------
            compiler.setCompileFlags(compilerOptions); // 将选项设置回编译器对象

            // 生成库文件
            dynamicLib_ = p.createDynamicLibrary(compiler, false);
            if (dynamicLib_) {
            std::cout << "[INFO] Library compiled and loaded successfully." << std::endl;
            } else {
                // 如果 dynamicLib_ 为空，说明内部 g++ 编译失败了
                std::cerr << "[FATAL ERROR] Internal G++ compilation failed!" << std::endl;
                std::cerr << "Hint: Check if the '-I' paths in compilerOptions match your directory structure." << std::endl;
            }
        }

        /**
         * 运行阶段：加载生成的库并计算
         */
        void compute(const std::vector<scalar_t>& x){
            // [修复] 直接使用内存中已加载的库，无需从磁盘手动加载
/*             if (!dynamicLib_) {
                std::cerr << "[ERROR] Library not loaded!" << std::endl;
                return;
            } */
           CppAD::cg::LinuxDynamicLib<scalar_t> library("./cppad_cg_model.so");
            auto model = library.model(funcADName_);

            // 6. 输入变量进行计算 (Execution)
            std::vector<double> y_out = model->ForwardZero(x);
            std::vector<double> jac_out = model->Jacobian(x);

            size_t n = x.size(); // 获取输入维度，修复 n 未定义的问题
            std::cout << "--- Result ---" << std::endl;
            std::cout << "Input:  [" << x[0] << ", " << x[1] << "]" << std::endl;
            std::cout << "Output: [" << y_out[0] << ", " << y_out[1] << "]" << std::endl;
            std::cout << "Jacobian Size: " << jac_out.size() << std::endl;
            std::cout << "Jacobian (dy0/dx0, dy0/dx1): {" 
              << jac_out[0 * n + 0] << ", " << jac_out[0 * n + 1] << "}" << std::endl;
        }
};

int main(){
    FuncAD funcADDemo("funcADDemo");

    std::vector<scalar_t> x_in = {std::numbers::pi / 6.0, 1.0};
    funcADDemo.compute(x_in);

    return 0;
}
