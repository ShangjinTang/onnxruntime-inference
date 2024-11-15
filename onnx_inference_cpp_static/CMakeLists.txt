cmake_minimum_required(VERSION 3.16)
project(onnxruntime_inference_cpp_static)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cc)

# copy model file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xgbc_iris.onnx
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.so)
target_include_directories(${PROJECT_NAME} PUBLIC onnxruntime/include)
