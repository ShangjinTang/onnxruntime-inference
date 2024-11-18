cmake_minimum_required(VERSION 3.16)
project(onnxruntime_inference_cpp_cmake)

set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} main.cpp onnx_inference_engine.cpp
                               onnx_inference_engine.h)

include(cmake/onnxruntime.cmake)

target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_HEADER_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIBRARY})

# copy model file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xgbc_iris.onnx
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
