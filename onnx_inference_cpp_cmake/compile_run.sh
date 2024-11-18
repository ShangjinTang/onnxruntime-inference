#!/usr/bin/env bash

set -e

if [[ -f ../xgboost_train/xgbc_iris.onnx ]]; then
    cp ../xgboost_train/xgbc_iris.onnx .
fi

rm -rf build &> /dev/null
cmake -S . -B build/Debug -DCMAKE_BUILD_TYPE=Debug && cp -f build/Debug/compile_commands.json . &> /dev/null && cmake --build build/Debug --config Debug
./build/Debug/onnxruntime_inference_cpp_cmake
