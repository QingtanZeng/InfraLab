#include <iostream>
#include <vector>

#include <cmath>

#include <thread>
#include <atomic>

int g_counter = 0;

std::atomic<int> g_atomic_counter(0);

void worker(){
    for(size_t i=0; i < 10000000; ++i){
        g_counter++;
        g_atomic_counter++;
    }
}

int main(){
    std::thread t1(worker);
    std::thread t2(worker);

    t1.join();
    t2.join();

    std::cout << g_counter << std::endl;
    std::cout << g_atomic_counter << std::endl;

    return 0;   
}

