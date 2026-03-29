#include <iostream>
#include <vector>
#include <cmath>
#include <numbers> // C++20 引入
#include <cppad/cppad.hpp>

int main(){
    using CppAD::AD;
    using CppAD::ADFun;

    size_t n=2;
    size_t m=2;
    std::vector<double> x(n);
    x[0]= std::numbers::pi / 6.0; x[1]=1.0;

    std::vector<AD<double>> ax(n);
    for(size_t i=0; i<n; i++){
        ax[i]=x[i];
    }
    CppAD::Independent(ax);

    std::vector<AD<double>> ay(m);
    ay[0] = ax[0]+ax[0]*ax[1];
    ay[1] = ax[1]*sin(ax[0]);

    CppAD::ADFun<double> f(ax, ay);

    std::vector<double> y = f.Forward(0, x); // 计算函数值 y
    std::vector<double> jac = f.Jacobian(x);


    std::cout << "Input x: {" << x[0] << ", " << x[1] << "}" << std::endl;
    std::cout << "Output y: {" << y[0] << ", " << y[1] << "}" << std::endl;
    std::cout << "Jacobian (dy0/dx0, dy0/dx1): {" 
              << jac[0 * n + 0] << ", " << jac[0 * n + 1] << "}" << std::endl;

    return 0;
}