#include <dlfcn.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "onnxruntime_cxx_api.h"

int main() {
    const char* model_path = "../xgboost_train/xgbc_iris.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Cpp_API");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    // Set the input and output names
    // Currently we use `onnx (python lib)` or `netron (online tool)` to get the node names.
    // TODO: check if can be read using C/C++ API without external tools.
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output_label", "output_probability"};

    // Define a 2D array for test samples
    // clang-format off
    std::vector<std::vector<float>> input_data_2d = {
        {6.1, 2.8, 4.7, 1.2},
        {5.7, 3.8, 1.7, 0.3},
        {7.7, 2.6, 6.9, 2.3},
        {6.0, 2.9, 4.5, 1.5},
        {6.8, 2.8, 4.8, 1.4},
        {5.4, 3.4, 1.5, 0.4},
        {5.6, 2.9, 3.6, 1.3},
        {6.9, 3.1, 5.1, 2.3},
        {6.2, 2.2, 4.5, 1.5},
        {5.8, 2.7, 3.9, 1.2},
        {6.5, 3.2, 5.1, 2.0},
        {4.8, 3.0, 1.4, 0.1},
        {5.5, 3.5, 1.3, 0.2},
        {4.9, 3.1, 1.5, 0.1},
        {5.1, 3.8, 1.5, 0.3},
        {6.3, 3.3, 4.7, 1.6},
        {6.5, 3.0, 5.8, 2.2},
        {5.6, 2.5, 3.9, 1.1},
        {5.7, 2.8, 4.5, 1.3},
        {6.4, 2.8, 5.6, 2.2},
        {4.7, 3.2, 1.6, 0.2},
        {6.1, 3.0, 4.9, 1.8},
        {5.0, 3.4, 1.6, 0.4},
        {6.4, 2.8, 5.6, 2.1},
        {7.9, 3.8, 6.4, 2.0},
        {6.7, 3.0, 5.2, 2.3},
        {6.7, 2.5, 5.8, 1.8},
        {6.8, 3.2, 5.9, 2.3},
        {4.8, 3.0, 1.4, 0.3},
        {4.8, 3.1, 1.6, 0.2}
    };
    // clang-format on

    // Flatten the 2D array into a 1D vector
    std::vector<float> input_data;
    for (const auto& row : input_data_2d) {
        input_data.insert(input_data.end(), row.begin(), row.end());
    }

    // Create a vector to hold the input tensor with shape [n, 4]
    int n = input_data_2d.size();

    // Create a vector to hold the input tensor with shape [n, 4]
    std::vector<int64_t> input_shape = {n, 4};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                              input_shape.data(), input_shape.size());

    // Perform inference
    std::vector<Ort::Value> output_tensors =
            session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(),
                        output_node_names.size());

    // Process the output
    Ort::Value& output_tensor = output_tensors.front();
    // Caution: be careful with the output tensor type, otherwise might get error values.
    int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();

    // Print the output
    for (size_t i = 0; i < output_tensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        std::cout << "Input: ";
        for (size_t j = 0; j < input_data_2d[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(1) << input_data_2d[i][j];
            if (j < input_data_2d.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " | Output: " << output_data[i] << std::endl;
    }

    session.release();
    session_options.release();
    env.release();

    return 0;
}
