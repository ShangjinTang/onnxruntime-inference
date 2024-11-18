#include "onnxruntime_cxx_api.h"
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>


#ifndef CPP_DEMO_ONNXINFERENCEENGINE_H
#define CPP_DEMO_ONNXINFERENCEENGINE_H


class OnnxInferenceEngine {
public:
    explicit OnnxInferenceEngine(const std::string &model_path)
            : env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_Cpp_API"),
              session_(env_, model_path.c_str(), Ort::SessionOptions{}) {
        session_options_.DisableCpuMemArena();
    }

    void setInput2D(const std::vector<std::vector<float>> &input_data_2d);

    void runInference();

    void printResults() const;

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions session_options_;

    std::vector<float> input_data_;
    std::vector<int64_t> input_shape_;
    Ort::Value input_tensor_{nullptr};

    std::vector<Ort::Value> output_tensors_;
    int n_{}; // Number of samples
    int m_{}; // Number of dimensions
};


#endif //CPP_DEMO_ONNXINFERENCEENGINE_H
