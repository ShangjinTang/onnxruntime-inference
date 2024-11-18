include(FetchContent)

# disable warning of FetchContent
if(POLICY CMP0135)
  message(VERBOSE "set policy cmp0135 to disable warning of FetchContent")
  cmake_policy(SET CMP0135 NEW)
endif()

set(TGZ_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz"
)

FetchContent_Declare(onnxruntime URL ${TGZ_URL})

FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_HEADER_DIR ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIBRARY ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
