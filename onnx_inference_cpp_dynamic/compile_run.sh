#!/usr/bin/env bash

set -e

if [[ -d ./onnxruntime ]]; then
    echo "./onnxruntime already exists."
else
    echo "Downloading onnxruntime ..."
    rm onnxruntime-linux-x64-1.20.0.tgz* || true
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
    tar xf onnxruntime-linux-x64-1.20.0.tgz
    mv onnxruntime-linux-x64-1.20.0 onnxruntime
    rm onnxruntime-linux-x64-1.20.0.tgz
fi

if [[ -f ../xgboost_train/xgbc_iris.onnx ]]; then
    cp ../xgboost_train/xgbc_iris.onnx .
fi

rm -rf build &> /dev/null
cmake -S . -B build/Debug -DCMAKE_BUILD_TYPE=Debug && cp -f build/Debug/compile_commands.json . &> /dev/null && cmake --build build/Debug --config Debug
cmake --build build/Debug
./build/Debug/onnxruntime_inference_cpp_dynamic
