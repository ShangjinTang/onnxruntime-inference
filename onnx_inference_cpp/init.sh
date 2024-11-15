#!/usr/bin/env bash

set -e

if [[ -d ./onnxruntime ]]; then
    echo "./onnxruntime already exists."
    exit
fi

echo "Downloading onnxruntime ..."
rm onnxruntime-linux-x64-1.20.0.tgz* || true
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar xf onnxruntime-linux-x64-1.20.0.tgz
mv onnxruntime-linux-x64-1.20.0 onnxruntime
rm onnxruntime-linux-x64-1.20.0.tgz

if [[ -f ../xgboost_train/xgbc_iris.onnx ]]; then
    cp ../xgboost_train/xgbc_iris.onnx .
fi
