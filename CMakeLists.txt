# 1. 声明 CMake 最低版本要求 (C++20/23 模块与特性支持推荐 CMake 3.28+)
cmake_minimum_required(VERSION 3.28)

# 默认优先寻找 LLVM-18 编译器 (clang-18 / clang++-18)，支持外部通过 -DCMAKE_CXX_COMPILER 或环境变量覆盖
if(NOT CMAKE_C_COMPILER)
    find_program(CLANG_C_COMPILER NAMES clang-18 clang)
    if(CLANG_C_COMPILER)
        set(CMAKE_C_COMPILER ${CLANG_C_COMPILER} CACHE FILEPATH "C Compiler")
    endif()
endif()

if(NOT CMAKE_CXX_COMPILER)
    find_program(CLANG_CXX_COMPILER NAMES clang++-18 clang++)
    if(CLANG_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER ${CLANG_CXX_COMPILER} CACHE FILEPATH "C++ Compiler")
    endif()
endif()

# 2. 声明项目名称 and 支持的语言
project(CTrjGen VERSION 1.0 LANGUAGES C CXX)

# 设置默认构建类型为 Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# ==========================================
# 核心指令：强制 CMake 导出 compile_commands.json
# ==========================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 3. 全局设置 C++ 标准为 C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # 禁用编译器特定扩展，保证可移植性

# 指定使用 LLVM 的标准库 libc++ (针对 C++ 编译与链接)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options($<$<LINK_LANGUAGE:CXX>:-stdlib=libc++>)

# 可选：使用 LLVM lld 链接器
find_program(LLD_EXECUTABLE NAMES lld-18 lld)
if(LLD_EXECUTABLE)
    add_link_options("-fuse-ld=lld")
endif()

# ==========================================
# 4. 构建目标
# ==========================================
# LDLT 分解目标 (NmrcCalc/LAPACK/LDLT.cpp)
add_executable(ldlt NmrcCalc/LAPACK/LDLT.cpp)
