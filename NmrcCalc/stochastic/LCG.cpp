#include <cstdint>
#include <iostream>
#include <random>
#include <concepts>

class LCG_M31{
public:
    // URBG 概念及 <random> 库分布器要求必须有 result_type 且为无符号整数
    using result_type = std::uint32_t;
    static constexpr result_type default_seed = 1u; // 推荐的标准随机数引擎默认种子

private:
    using scalar = std::int32_t;
    static constexpr scalar A = 16807;  //7^5
    static constexpr scalar M = 2147483647;     //2^31 -1 
    static constexpr scalar Q = 127773;
    static constexpr scalar R = 2836;

    scalar state_;

public:
    // URBG concepts: define the maximum and minimum values
    [[nodiscard]] static constexpr result_type min() noexcept { return 1u; }
    [[nodiscard]] static constexpr result_type max() noexcept { return static_cast<result_type>(M - 1); }

    // constructor, support initialization in build
    constexpr explicit LCG_M31(result_type seedvalue = default_seed){
        seed(seedvalue);
    }

    // set seed
    // 标准引擎对外通常接受 result_type 作为种子类型
    constexpr void seed(result_type seedvalue = default_seed){
        if(seedvalue == 0 || seedvalue >= static_cast<result_type>(M)){
            state_ = static_cast<scalar>(default_seed);
        }else{
            state_ = static_cast<scalar>(seedvalue);
        }
    }

    // Overload operator() to generate random value
    constexpr result_type operator()(){

        scalar L = state_ % Q;     // integer remainder
        scalar K = state_ / Q;     // integer division
        
        scalar temp = A*L - R*K;

        // (al-rk)(mod m)
        if(temp <= 0 )
            state_ = temp + M;
        else
            state_ = temp;

        return static_cast<result_type>(state_);
    }
    
    // 满足 URBG 概念：比较运算符
    friend constexpr bool operator==(const LCG_M31& lhs, const LCG_M31& rhs) {
        return lhs.state_ == rhs.state_;
    }
};

// C++20 URBG 断言应放在类完整定义之后，并且补齐遗漏的 ">"
static_assert(std::uniform_random_bit_generator<LCG_M31>);

int main(){
    // 1. 基础测试：生成原始的 LCG 整数序列
    LCG_M31 lcg(12345); // 设置初始种子
    std::cout << "原始 LCG 输出 (前 5 个):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << lcg() << std::endl;
    }

    std::cout << "\n-------------------\n\n";

    // 2. 结合 C++ <random> 库生成 [0, 1) 的浮点数
    // 因为我们的类满足 std::uniform_random_bit_generator 概念，所以可以直接传给分布器
    LCG_M31 lcg_float(98765);
    std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
    
    std::cout << "[0, 1) 均匀分布浮点数 (前 5 个):" << std::endl;
    for (int i = 0; i < 50; ++i) {
        std::cout << float_dist(lcg_float) << std::endl;
    }

    std::cout << "\n-------------------\n\n";

    // 3. C++20 特性：编译期生成随机数 (Compile-time evaluation)
    // 借助 constexpr，我们可以在编译期推导出某个种子经过 N 次迭代后的状态
    constexpr auto compile_time_rand = []() consteval {
        LCG_M31 ct_lcg(1);
        ct_lcg(); ct_lcg(); // 迭代两次
        return ct_lcg();    // 返回第三次的值
    }();

    std::cout << "编译期计算的第 3 个随机数: " << compile_time_rand << std::endl;

    return 0;
}
