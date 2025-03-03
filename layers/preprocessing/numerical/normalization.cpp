#include "layers/preprocessing/numerical/normalization.h"
#include <fstream>
#include <cmath>
#include <stdexcept>

NormalizationLayer::NormalizationLayer(const std::vector<size_t>& input_shape, float epsilon)
    : input_shape(input_shape), mean(Tensor()), variance(Tensor()), epsilon(epsilon), input_cache(Tensor()) {
    if (input_shape.empty()) {
        throw std::invalid_argument("NormalizationLayer input shape cannot be empty");
    }
    if (epsilon <= 0.0f) {
        throw std::invalid_argument("Epsilon must be positive");
    }
}

Tensor NormalizationLayer::forward(const Tensor& input) {
    if (input.shape() != input_shape) {
        throw std::invalid_argument("Input shape does not match expected shape in NormalizationLayer");
    }
    if (mean.size() == 0 || variance.size() == 0) {
        throw std::runtime_error("NormalizationLayer must be adapted before use");
    }
    input_cache = input;
    Tensor output(input_shape);
    float* output_data = output.data();
    const float* input_data = input.data();
    const float* mean_data = mean.data();
    const float* var_data = variance.data();
    size_t size = input.size();
    size_t feature_size = mean.size();

    for (size_t i = 0; i < size; ++i) {
        size_t feature_idx = i % feature_size;
        output_data[i] = (input_data[i] - mean_data[feature_idx]) / std::sqrt(var_data[feature_idx] + epsilon);
    }

    return output;
}

Tensor NormalizationLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_shape) {
        throw std::invalid_argument("Gradient shape must match input shape in NormalizationLayer");
    }
    Tensor grad_input(input_shape);
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* var_data = variance.data();
    size_t size = grad_output.size();
    size_t feature_size = variance.size();

    for (size_t i = 0; i < size; ++i) {
        size_t feature_idx = i % feature_size;
        grad_input_data[i] = grad_output_data[i] / std::sqrt(var_data[feature_idx] + epsilon);
    }

    return grad_input;
}

void NormalizationLayer::adapt(const Tensor& data) {
    if (data.shape().size() < 2 || data.shape()[0] == 0) {
        throw std::invalid_argument("Data must be at least 2D with non-zero batch size for adaptation");
    }
    computeStatistics(data);
}

void NormalizationLayer::computeStatistics(const Tensor& data) {
    size_t batch_size = data.shape()[0];
    size_t feature_size = data.size() / batch_size;
    mean = Tensor({feature_size});
    variance = Tensor({feature_size});
    float* mean_data = mean.data();
    float* var_data = variance.data();
    const float* data_ptr = data.data();

    // Compute mean
    for (size_t f = 0; f < feature_size; ++f) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += data_ptr[b * feature_size + f];
        }
        mean_data[f] = sum / batch_size;
    }

    // Compute variance
    for (size_t f = 0; f < feature_size; ++f) {
        float sum_sq = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            float diff = data_ptr[b * feature_size + f] - mean_data[f];
            sum_sq += diff * diff;
        }
        var_data[f] = sum_sq / (batch_size - 1); // Sample variance
    }
}

void NormalizationLayer::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    file.write(reinterpret_cast<const char*>(mean.data()), mean.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(variance.data()), variance.size() * sizeof(float));
    size_t cache_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&cache_size), sizeof(cache_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), cache_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<NormalizationLayer> NormalizationLayer::load(std::ifstream& file) {
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> input_shape(shape_size);
    file.read(reinterpret_cast<char*>(input_shape.data()), shape_size * sizeof(size_t));
    float epsilon;
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    auto layer = std::make_unique<NormalizationLayer>(input_shape, epsilon);

    size_t feature_size = input_shape[1]; // Assuming {batch_size, features, ...}
    for (size_t i = 2; i < input_shape.size(); ++i) {
        feature_size *= input_shape[i];
    }
    layer->mean = Tensor({feature_size});
    layer->variance = Tensor({feature_size});
    file.read(reinterpret_cast<char*>(layer->mean.data()), layer->mean.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->variance.data()), layer->variance.size() * sizeof(float));

    size_t cache_size;
    file.read(reinterpret_cast<char*>(&cache_size), sizeof(cache_size));
    std::vector<size_t> cache_shape(cache_size);
    file.read(reinterpret_cast<char*>(cache_shape.data()), cache_size * sizeof(size_t));
    layer->input_cache = Tensor(cache_shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));

    return layer;
}

void NormalizationLayer::print() const {
    std::cout << "NormalizationLayer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
    std::cout << "Epsilon: " << epsilon << "\n";
    std::cout << "Mean:\n";
    mean.print();
    std::cout << "Variance:\n";
    variance.print();
    std::cout << "Input Cache:\n";
    input_cache.print();
}
