#!/usr/bin/env bash

unset LD_LIBRARY_PATH &&
    clang++ -O0 -g -std=c++20 -Wall -Wextra -Wpedantic -ldl -o a.out main_dynamic.cc -I./onnxruntime/include/ && ./a.out
