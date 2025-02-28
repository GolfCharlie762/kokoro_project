#include "core/preprocess.h"
#include <numeric>
#include <sstream>

///Preprocessing tests////////////////////
//// Test 1: Numerical Data Preprocessing
//std::cout << "Testing Numerical Data Preprocessing\n";
//std::vector<float> num_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
//Tensor num_tensor({1, 5}, num_data);

//std::cout << "Original Data:\n";
//num_tensor.print();

//Tensor standardized = DataPreprocessor::standardize(num_tensor);
//std::cout << "Standardized:\n";
//standardized.print();

//Tensor normalized = DataPreprocessor::normalize(num_tensor, 0.0f, 1.0f);
//std::cout << "Normalized [0, 1]:\n";
//normalized.print();

//Tensor clipped = DataPreprocessor::clip(num_tensor, 2.0f, 4.0f);
//std::cout << "Clipped [2, 4]:\n";
//clipped.print();

//Tensor padded = DataPreprocessor::pad(num_tensor, {1, 7}, 0.0f);
//std::cout << "Padded to {1, 7}:\n";
//padded.print();

//Tensor one_hot = DataPreprocessor::oneHotEncode(Tensor({1, 3}, {0.0f, 1.0f, 2.0f}), 3);
//std::cout << "One-Hot Encoded (3 classes):\n";
//one_hot.print();

//Tensor resized = DataPreprocessor::resize(num_tensor, {1, 3});
//std::cout << "Resized to {1, 3}:\n";
//resized.print();

//// Test 2: Text Preprocessing
//std::cout << "\nTesting Text Preprocessing\n";
//std::string text = "hello world this is a test";
//std::vector<std::string> tokens = DataPreprocessor::tokenizeText(text);
//std::cout << "Tokens: ";
//for (const auto& token : tokens) std::cout << token << " ";
//std::cout << "\n";

//std::vector<std::string> texts = {"hello world", "this is test", "hello test"};
//std::map<std::string, size_t> vocab = DataPreprocessor::buildVocabulary(texts);
//std::cout << "Vocabulary: ";
//for (const auto& pair : vocab) std::cout << pair.first << ":" << pair.second << " ";
//std::cout << "\n";

//Tensor text_tensor = DataPreprocessor::textToTensor(tokens, vocab);
//std::cout << "Text Tensor:\n";
//text_tensor.print();

//// Test 3: Image Preprocessing
//std::cout << "\nTesting Image Preprocessing\n";
//std::vector<float> img_data = {
//    255.0f, 0.0f, 128.0f,   // RGB pixel 1
//    100.0f, 150.0f, 200.0f  // RGB pixel 2
//};
//Tensor image({2, 1, 3}, img_data); // {height, width, channels}
//std::cout << "Original Image:\n";
//image.print();

//Tensor gray = DataPreprocessor::grayscale(image);
//std::cout << "Grayscale Image:\n";
//gray.print();

//Tensor norm_img = DataPreprocessor::normalizeImage(image, 0.0f, 255.0f);
//std::cout << "Normalized Image:\n";
//norm_img.print();

//Tensor cropped = DataPreprocessor::cropImage(image, 0, 0, 1, 1);
//std::cout << "Cropped Image:\n";
//cropped.print();

//// Test 4: Statistical Functions
//std::cout << "\nTesting Statistical Functions\n";
//std::cout << "Mean: " << DataPreprocessor::mean(num_tensor) << "\n";
//float mean_val = DataPreprocessor::mean(num_tensor);
//std::cout << "Variance: " << DataPreprocessor::variance(num_tensor, mean_val) << "\n";
//std::cout << "StdDev: " << DataPreprocessor::stddev(num_tensor, mean_val) << "\n";

Tensor DataPreprocessor::standardize(const Tensor& data) {
    Tensor result(data.shape());
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t size = data.size();
    float mean_val = mean(data);
    float std_val = stddev(data, mean_val);
    if (std_val == 0.0f) return result;
    float inv_std = 1.0f / std_val;
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = (data_ptr[i] - mean_val) * inv_std;
    }
    return result;
}

Tensor DataPreprocessor::normalize(const Tensor& data, float min_val, float max_val) {
    Tensor result(data.shape());
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t size = data.size();
    float data_min = *std::min_element(data_ptr, data_ptr + size);
    float data_max = *std::max_element(data_ptr, data_ptr + size);
    if (data_max == data_min) return result;
    float scale = (max_val - min_val) / (data_max - data_min);
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = min_val + (data_ptr[i] - data_min) * scale;
    }
    return result;
}

Tensor DataPreprocessor::clip(const Tensor& data, float min_val, float max_val) {
    Tensor result(data.shape());
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t size = data.size();
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(min_val, std::min(max_val, data_ptr[i]));
    }
    return result;
}

Tensor DataPreprocessor::pad(const Tensor& data, const std::vector<size_t>& target_shape, float pad_value) {
    if (data.shape().size() != target_shape.size()) {
        throw std::invalid_argument("Data and target shape dimensions must match");
    }
    Tensor result(target_shape, pad_value);
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t dims = data.shape().size();
    std::vector<size_t> copy_shape = data.shape();
    for (size_t i = 0; i < dims; ++i) {
        copy_shape[i] = std::min(data.shape()[i], target_shape[i]);
    }
    for (size_t i = 0; i < copy_shape[0]; ++i) {
        for (size_t j = 0; j < copy_shape[1]; ++j) {
            size_t src_idx = i * data.shape()[1] + j;
            size_t dst_idx = i * target_shape[1] + j;
            if (dims > 2) {
                for (size_t k = 0; k < copy_shape[2]; ++k) {
                    src_idx = (src_idx * data.shape()[2]) + k;
                    dst_idx = (dst_idx * target_shape[2]) + k;
                    result_data[dst_idx] = data_ptr[src_idx];
                }
            } else {
                result_data[dst_idx] = data_ptr[src_idx];
            }
        }
    }
    return result;
}

Tensor DataPreprocessor::oneHotEncode(const Tensor& data, size_t num_classes) {
    if (data.shape().size() != 2) {
        throw std::invalid_argument("One-hot encoding expects 2D input {batch_size, sequence_length}");
    }
    size_t batch_size = data.shape()[0];
    size_t seq_length = data.shape()[1];
    Tensor result({batch_size, seq_length, num_classes});
    float* result_data = result.data();
    const float* data_ptr = data.data();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            size_t idx = static_cast<size_t>(data_ptr[b * seq_length + t]);
            if (idx >= num_classes) {
                throw std::out_of_range("Index exceeds number of classes");
            }
            result_data[(b * seq_length + t) * num_classes + idx] = 1.0f;
        }
    }
    return result;
}

Tensor DataPreprocessor::resize(const Tensor& data, const std::vector<size_t>& new_shape) {
    if (data.shape().size() != new_shape.size()) {
        throw std::invalid_argument("Data and new shape dimensions must match");
    }
    Tensor result(new_shape);
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t old_size = data.size();
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    size_t copy_size = std::min(old_size, new_size);
    std::copy(data_ptr, data_ptr + copy_size, result_data);
    if (new_size > old_size) {
        std::fill(result_data + old_size, result_data + new_size, 0.0f);
    }
    return result;
}

std::vector<std::string> DataPreprocessor::tokenizeText(const std::string& text, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

Tensor DataPreprocessor::textToTensor(const std::vector<std::string>& tokens, const std::map<std::string, size_t>& vocab) {
    Tensor result({1, tokens.size()});
    float* result_data = result.data();
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = vocab.find(tokens[i]);
        result_data[i] = (it != vocab.end()) ? static_cast<float>(it->second) : 0.0f;
    }
    return result;
}

std::map<std::string, size_t> DataPreprocessor::buildVocabulary(const std::vector<std::string>& texts) {
    std::map<std::string, size_t> vocab;
    size_t index = 1; // Reserve 0 for unknown/OOV
    for (const auto& text : texts) {
        auto tokens = tokenizeText(text);
        for (const auto& token : tokens) {
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = index++;
            }
        }
    }
    return vocab;
}

Tensor DataPreprocessor::grayscale(const Tensor& image) {
    if (image.shape().size() != 3 || image.shape()[2] != 3) {
        throw std::invalid_argument("Grayscale expects 3D image tensor with 3 channels {height, width, 3}");
    }
    size_t height = image.shape()[0];
    size_t width = image.shape()[1];
    Tensor result({height, width, 1});
    float* result_data = result.data();
    const float* image_data = image.data();
    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            size_t idx = (h * width + w) * 3;
            result_data[h * width + w] = 0.299f * image_data[idx] + 0.587f * image_data[idx + 1] + 0.114f * image_data[idx + 2];
        }
    }
    return result;
}

Tensor DataPreprocessor::normalizeImage(const Tensor& image, float mean, float std) {
    Tensor result(image.shape());
    float* result_data = result.data();
    const float* image_data = image.data();
    size_t size = image.size();
    float inv_std = 1.0f / std;
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = (image_data[i] - mean) * inv_std;
    }
    return result;
}

Tensor DataPreprocessor::cropImage(const Tensor& image, size_t start_h, size_t start_w, size_t height, size_t width) {
    if (image.shape().size() < 2 || start_h + height > image.shape()[0] || start_w + width > image.shape()[1]) {
        throw std::invalid_argument("Invalid crop dimensions or shape");
    }
    size_t channels = (image.shape().size() == 3) ? image.shape()[2] : 1;
    std::vector<size_t> new_shape = {height, width};
    if (channels > 1) new_shape.push_back(channels);
    Tensor result(new_shape);
    float* result_data = result.data();
    const float* image_data = image.data();
    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            size_t src_idx = (start_h + h) * image.shape()[1] + (start_w + w);
            size_t dst_idx = h * width + w;
            if (channels == 1) {
                result_data[dst_idx] = image_data[src_idx];
            } else {
                for (size_t c = 0; c < channels; ++c) {
                    result_data[dst_idx * channels + c] = image_data[src_idx * channels + c];
                }
            }
        }
    }
    return result;
}

float DataPreprocessor::mean(const Tensor& data) {
    if (data.size() == 0) return 0.0f;
    const float* data_ptr = data.data();
    return std::accumulate(data_ptr, data_ptr + data.size(), 0.0f) / data.size();
}

float DataPreprocessor::variance(const Tensor& data, float mean_val) {
    if (data.size() <= 1) return 0.0f;
    const float* data_ptr = data.data();
    float sum_sq = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        float diff = data_ptr[i] - mean_val;
        sum_sq += diff * diff;
    }
    return sum_sq / (data.size() - 1);
}

float DataPreprocessor::stddev(const Tensor& data, float mean_val) {
    return std::sqrt(variance(data, mean_val));
}
