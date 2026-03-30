#!/bin/bash

# 定义日志文件
LOG_FILE="build.log"

# 清空旧日志
echo "Starting build at $(date)" > $LOG_FILE

# 执行编译命令
# 注意：这里使用 "$@" 接收传入的文件名参数，或者你也可以写死 main.cpp
/usr/bin/g++ \
    -fdiagnostics-color=always \
    -g \
    -std=c++23 \
    autodiff_codegen_demo.cpp \
    -o autodiff_codegen_demo \
    -I./lib/cppad/install/include \
    -I./lib/cppadcodegen/install/include \
    -I/usr/include/eigen3 \
    -L./lib/cppad/install/lib \
    -lcppad_lib \
    -ldl \
    -Wl,-rpath,./lib/cppad/install/lib \
    2>&1 | tee -a $LOG_FILE

# 检查编译结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Build Success!" | tee -a $LOG_FILE
else
    echo "Build Failed!" | tee -a $LOG_FILE
    exit 1
fi