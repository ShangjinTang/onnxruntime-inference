#include <dlfcn.h>

#include <iomanip>
#include <iostream>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

using GetApiBaseFunc = OrtApiBase* (*)();

class OnnxInferenceEngine {
public:
    explicit OnnxInferenceEngine(const std::string& model_path)
            : env_(ORT_LOGGING_LEVEL_WARNING, "ONNX_Cpp_API"),
              session_(env_, model_path.c_str(), Ort::SessionOptions{}) {
        session_options_.DisableCpuMemArena();
    }

    void setInput2D(const std::vector<std::vector<float>>& input_data_2d) {
        // Flatten the 2D array into a 1D vector
        input_data_.clear();
        for (const auto& row : input_data_2d) {
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

    void runInference() {
        std::array<const char*, 1> input_names = {"input"};
        std::array<const char*, 2> output_names = {"output_label", "output_probability"};

        // Perform inference
        output_tensors_ = session_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor_, input_names.size(),
                                     output_names.data(), output_names.size());
    }

    void printResults() const {
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


int main() {
    sleep(3);

    const char* model_path = "./xgbc_iris.onnx";

    // NOTE: It's better to use absolute link in dlopen. Relative link below is only for demostration.
    void* ort_library = dlopen("./onnxruntime/lib/libonnxruntime.so.1", RTLD_LAZY);
    if (ort_library == nullptr) {
        std::cerr << "Failed to load libonnxruntime.so: " << dlerror() << '\n';
        return 1;
    }

    auto get_api_base = reinterpret_cast<GetApiBaseFunc>(dlsym(ort_library, "OrtGetApiBase"));
    if (get_api_base == nullptr) {
        dlclose(ort_library);
        return 1;
    }

    const OrtApi* api = get_api_base()->GetApi(ORT_API_VERSION);

    Ort::InitApi(api);

    {
        // Create an instance of OnnxInferenceEngine
        OnnxInferenceEngine classifier(model_path);

        // Define test samples (input data)
        const std::vector<std::vector<float>> input_data_2d = {
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

        // Set input data and run inference
        classifier.setInput2D(input_data_2d);
        classifier.runInference();

        // Print the results
        classifier.printResults();
    }

    sleep(3);

    const int result = dlclose(ort_library);
    if (result != 0) {
        std::cerr << "Failed to call dlclose()" << '\n';
        return 1;
    }

    sleep(3);

    return 0;
}
