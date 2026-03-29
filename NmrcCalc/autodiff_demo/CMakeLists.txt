cmake_minimum_required(VERSION 3.10)
project(AutoDiff_Demo)

set(CPPAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/cppad/include")

add_executable(autodiff_demo autodiff_demo.cpp)

target_include_directories(autodiff_demo PRIVATE ${CPPAD_INCLUDE_DIR})