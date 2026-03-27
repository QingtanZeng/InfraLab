#include <iostream>
#include <vector>

#include <cmath>
#include <numeric>

#include <thread>
#include <atomic>
#include <new>

#include <cassert>


// 模拟复杂的动力学数据结构
struct DynamicsData {
    double mass_matrix[6][6];
    double bias_vector[6];
};

//Cache line vector
#ifdef __cpp_lib_hardware_interference_size
    constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE_SIZE = 64;
#endif

struct alignas(CACHE_LINE_SIZE) ThreadLocalResult {
    double total_energy = 0.0;
    double max_violation = 0.0;
};

// Parallel propagation of dynamics
class TrajectoryPropagator {
public:
    double setupTrjPbm(const std::vector<double>& state_trajectory,
                        std::vector<DynamicsData>& dicsdyn){
    
        // check nodes number
        size_t N = state_trajectory.size();
        assert( N==dicsdyn.size() &&"Vector dimensions must match");
        
        // parallel core number
        const size_t num_threads = 4;

        // define local aligned structure for thread results
        std::vector<ThreadLocalResult> resultthreadloc(num_threads);

        // define atomic counter for dynamic load balancing
        std::atomic<size_t> atomic_idx{0};

        // define parallel calculation task
        auto discnode_task = [&](size_t worker_id){
            // refer thread's local result
            ThreadLocalResult& resthdloc = resultthreadloc[worker_id];

            size_t idx = atomic_idx.fetch_add(1, std::memory_order_relaxed);    // local value of atomic index

            while(idx<N){
                double state_val = state_trajectory[idx];
                double energy = state_val * state_val * 0.5 + std::sin(state_val);
                
                // 2. 写入全局结果 (使用 std::move 或直接写入避免拷贝)
                // 假设这是构建复杂的动力学矩阵
                for(int r=0; r<10; ++r) dicsdyn[idx].bias_vector[r] = energy * r * 0.001;

                resthdloc.total_energy += energy;

                idx = atomic_idx.fetch_add(1, std::memory_order_relaxed);
            }
        };

        // launch threads
        {
            std::vector<std::jthread> pool;
            pool.reserve(num_threads);
            for(size_t id=0; id < num_threads; ++id){
                pool.emplace_back(discnode_task, id);
            }
        }

        // aggregate thread-local results
        double final_total_energy = 0.0;
        for(const auto& res:resultthreadloc){
            final_total_energy += res.total_energy;
        }
        return final_total_energy;
    }
};

int main(){
    std::vector<double> trjRef(10000000, 2);
    std::vector<DynamicsData> dicsdyn(10000000);

    TrajectoryPropagator propagator;
    double total_cost = propagator.setupTrjPbm(trjRef, dicsdyn);

    std::cout << "Total Cost: " << total_cost << std::endl;
    // 验证部分数据
    std::cout << "Dynamics[0] Bias[1]: " << dicsdyn[0].bias_vector[1] << std::endl;
    
    return 0;
}