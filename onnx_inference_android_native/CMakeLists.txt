cmake_minimum_required(VERSION 3.0)

project(onnxruntime_main)

add_executable(${PROJECT_NAME} main.cc)

target_include_directories(${PROJECT_NAME} PUBLIC onnxruntime/headers/)
