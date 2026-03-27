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

#include <queue>
#include <unordered_map>
#include <algorithm>

// Constant Values
const double PI = 3.14159265358979323846;
const double STEERMAX = 0.6;    // 最大转角rad
const size_t STEERNUM = 5;      // 离散转向动作数量
const double STEP_SIZE = 1.0;   // 单步移动距离
const double XY_RES = 1.0;      // X/Y网格分辨率
const double YAW_RES = 0.1;     // 角度网格分辨率

const double WB = 2.5;      // Wheel base 轴距

// Define node
class Node {
public:
    double x, y, theta;     // 连续状态
    double g, f;            // 实际代价，总代价
    Node* parent;           // 父节点

    size_t idx_x, idx_y, idx_theta;    // 网格索引坐标

    Node(double _x, double _y, double _theta, double _g, double _h, Node* _p) 
        : x(_x), y(_y), theta(_theta), g(_g), f(_g +_h), parent(_p) {
        idx_x = std::round(_x / XY_RES);
        idx_y = std::round(_y / XY_RES);
        idx_theta = std::round(_theta / YAW_RES);
    }
};

// Comparator
class CompareNode{
public:
    bool operator()(const Node& lhs, const Node& rhs){
        return lhs.f > rhs.f;
    }
};

// 启发式函数(欧几里得距离，update: Reeds-shepp)
double calc_heuristic(double x, double y, double goal_x, double goal_y){
    return std::hypot(x-goal_x, y-goal_y);
}

// Motion Primitive
void update_state(double x, double y, double theta, double steer,
                    double& nx, double& ny, double& ntheta){
    nx = x + STEP_SIZE * std::cos(theta);
    ny = y + STEP_SIZE * std::sin(theta);
    ntheta = theta + (STEP_SIZE / WB) * std::tan(steer);
    
    while (ntheta > PI) ntheta -= 2.0 * PI;
    while (ntheta < -PI) ntheta += 2.0 * PI;
}

// 生成唯一索引用于Closed Set检查
long long generate_key(int x, int y, int theta) {
    return ((long long)x << 32) | ((long long)y << 16) | (theta + 1000); 
}

// Hybrid A* path planning implementation
std::vector<Node> hybridAStar(double start_x, double start_y, double start_theta,
                                double goal_x, double goal_y){
    std::priority_queue<Node*, std::vector<Node*>, CompareNode> open_set;
    std::unordered_map<long long, double> closed_set;

    Node* start_node = new Node(start_x, start_y, start_theta, 
                                0.0, calc_heuristic(start_x, start_y, goal_x, goal_y), nullptr);
    open_set.push(start_node);

    while(!open_set.empty()) {
        Node* current = open_set.top();
        open_set.pop();

        // check terminal condition
        if(std::hypot(current->x - goal_x, current->y - goal_y) < 1.0){
            std::cout << "Goal reached!" << std::endl;
            std::vector<Node> path;
            while(current != nullptr){
                path.push_back(*current);
                current = current->parent;
            }
            return path;
        }

        long long key = generate_key(current->idx_x, current->idx_y, current->idx_theta);
        // 剪枝
        if(closed_set.find(key) != closed_set.end() && closed_set[key] <= current->g){
            delete current;
            continue;
        }
        closed_set[key] = current->g;

        // 扩展节点
        for(size_t idx=0; idx < STEERNUM; idx++){
            double steering = -STEERMAX + (2.0 * STEERMAX * idx) / (STEERNUM - 1);
            double nx, ny, ntheta;
            update_state(current->x, current->y, current->theta, steering, nx, ny, ntheta);
            // TODO: 在此处添加 check_collision(nx, ny)
            bool collision = false;

            if (!collision) {
                double new_g = current->g + STEP_SIZE; // 这里简化代价为距离
                double new_h = calc_heuristic(nx, ny, goal_x, goal_y);
                
                Node* next_node = new Node(nx, ny, ntheta, new_g, new_h, current);
                
                // 再次检查新节点所在网格是否值得加入
                long long next_key = generate_key(next_node->idx_x, next_node->idx_y, next_node->idx_theta);
                if (closed_set.find(next_key) == closed_set.end() || closed_set[next_key] > new_g) {
                    open_set.push(next_node);
                } else {
                    delete next_node;
                }
            }
        }
    }
    return {}; // 未找到路径
}

// Main 
int main(){
    auto path = hybridAStar(0.0, 0.0, 0.0, 10.0, 10.0);
    std::cout << "Path nodes: " << path.size() << std::endl;
    return 0;
}