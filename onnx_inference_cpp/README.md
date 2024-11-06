# ONNX Runtime C++ Inference

## Download precompiled library

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar xf onnxruntime-linux-x64-1.20.0.tgz
mv onnxruntime-linux-x64-1.20.0.tgz onnxruntime
rm onnxruntime-linux-x64-1.20.0.tgz
```

## Run Inference

### Static

```bash
export LD_LIBRARY_PATH=./onnxruntime/lib
clang++ -O0 -g -std=c++20 -Wall -Wextra -Wpedantic -ldl -o a.out main_static.cc -I./onnxruntime/include/ -L./onnxruntime/lib -lonnxruntime && ./a.out
```

### Dynamic

```bash
unset LD_LIBRARY_PATH
clang++ -O0 -g -std=c++20 -Wall -Wextra -Wpedantic -ldl -o a.out main_dynamic.cc -I./onnxruntime/include/ && ./a.out
```
