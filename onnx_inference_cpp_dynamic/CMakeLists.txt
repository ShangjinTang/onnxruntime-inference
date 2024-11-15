cmake_minimum_required(VERSION 3.16)
project(onnxruntime_inference_cpp_dynamic)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cc
)

# copy runtime so
file(GLOB ONNXRUNTIME_LIBS
     "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.so*")

foreach(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIBS})
  file(COPY ${ONNXRUNTIME_LIB}
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib)
endforeach()

# copy model file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xgbc_iris.onnx
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${PROJECT_NAME} PUBLIC onnxruntime/include)
