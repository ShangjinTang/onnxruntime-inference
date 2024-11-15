#!/usr/bin/env bash

export LD_LIBRARY_PATH=./onnxruntime/lib &&
    clang++ -O0 -g -std=c++20 -Wall -Wextra -Wpedantic -ldl -o a.out main_static.cc -I./onnxruntime/include/ -L./onnxruntime/lib -lonnxruntime && ./a.out
