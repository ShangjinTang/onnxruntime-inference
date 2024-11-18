#include "onnx_inference_engine.h"

void OnnxInferenceEngine::setInput2D(const std::vector<std::vector<float>> &input_data_2d) {
    // Flatten the 2D array into a 1D vector
    input_data_.clear();
    for (const auto &row: input_data_2d) {
        input_data_.insert(input_data_.end(), row.begin(), row.end());
    }

    // Create a vector to hold the input tensor with shape [n, m]
    n_ = input_data_2d.size();
    m_ = input_data_2d[0].size();
    input_shape_ = {n_, m_};

    // Create the input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_data_.data(), input_data_.size(),
                                                    input_shape_.data(), input_shape_.size());
}

void OnnxInferenceEngine::runInference() {
    std::array<const char *, 1> input_names = {"input"};
    std::array<const char *, 2> output_names = {"output_label", "output_probability"};

    // Perform inference
    output_tensors_ = session_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor_, input_names.size(),
                                   output_names.data(), output_names.size());
}

void OnnxInferenceEngine::printResults() const {
    // Process the output
    auto& output_tensor = const_cast<Ort::Value &>(output_tensors_.front());
    // Caution: be careful with the output tensor type, otherwise might get error values.
    auto* output_data = output_tensor.GetTensorMutableData<int64_t>();

    for (size_t i = 0; i < output_tensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        std::cout << "Input: ";
        for (size_t j = 0; j < m_; ++j) {  // Assuming 4 features in each input sample
            std::cout << std::fixed << std::setprecision(1) << input_data_[(i * m_) + j]; // Adjust indexing
            if (j != m_) {
                std::cout << ", ";
            }
        }
        std::cout << " | Output Label: " << output_data[i] << '\n';
    }
}
