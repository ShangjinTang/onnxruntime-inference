# OnnxRuntime Android NDK Deployment

## Download Android NDK

1. Download Android NDK in `https://developer.android.com/ndk/downloads`, for example, `android-ndk-r27c-linux.zip`.

2. Unzip to `~/Android/Ndk`. Ensure `~/Android/Ndk/README.md` exists.

3. Add `export ANDROID_NDK=~/Android/Ndk` to shell rc (.bashrc or .zshrc), re-login shell.

## Download Precompiled OnnxRuntime Library

Reference: [Install ONNX Runtime](https://onnxruntime.ai/docs/install/)

- Download the [onnxruntime-training-android (full package)](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android) AAR hosted at Maven Central.
- Change the file extension from `.aar` to `.zip`, and unzip it.
- Include the header files from the `headers` folder.
- Include the relevant `libonnxruntime.so` dynamic library from the `jni` folder in your NDK project.

After above steps, the folder structure should be like this:

```text
 .
├──  onnxruntime
│  ├──  headers
│  │  ├──  onnxruntime_c_api.h
│  │  ├──  onnxruntime_cxx_api.h
│  │  └──  ...
│  ├──  jni
│  │  ├──  arm64-v8a
│  │  ├──  armeabi-v7a
│  │  ├──  x86
│  │  └──  x86_64
│  ├──  META-INF
│  ├── 󰗀 AndroidManifest.xml
│  ├──  classes.jar
│  └──  R.txt
├──  CMakeLists.txt
├──  main.cc
└──  README.md
```

## Build Executable

```bash
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-35 \
    -DANDROID_STL=c++_static \
    -S . -B build &&
    cmake --build build
```

Note: For deploying on android emulator, use `-DANDROID_ABI=x86_64`.

## Push

```bash
adb shell mkdir -p /data/local/tmp/onnxruntime_test/onnxruntime/lib

adb push build/onnxruntime_main /data/local/tmp/onnxruntime_test
adb push ../xgboost_train/xgbc_iris.onnx /data/local/tmp/onnxruntime_test
adb push onnxruntime/jni/arm64-v8a/libonnxruntime.so /data/local/tmp/onnxruntime_test/onnxruntime/lib/libonnxruntime.so.1
```

The structure should be as below:

```text
 .
├──  onnxruntime
│  └──  lib
│     └──  libonnxruntime.so.1
├──  onnxruntime_main
└──  xgbc_iris.onnx
```

# Run

```bash
adb shell "cd /data/local/tmp/onnxruntime_test && ./onnxruntime_main"
```
