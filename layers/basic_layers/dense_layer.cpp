#include "dense_layer.h"
#include <random>
#include <algorithm>
#include <fstream>
#include <cmath>

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size),
      weights({input_size, output_size}), biases({output_size}) {
    float limit = std::sqrt(6.0f / (input_size + output_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);
    std::generate(weights.data(), weights.data() + weights.size(), [&]() { return dis(gen); });
    biases.fill(0.0f);
}

Tensor DenseLayer::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != input_size) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, input_size}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    Tensor output({batch_size, output_size});
    float* output_data = output.data();
    const float* input_data = input.data();
    const float* weights_data = weights.data();
    const float* biases_data = biases.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t j = 0; j < output_size; ++j) {
            float sum = biases_data[j];
            for (size_t i = 0; i < input_size; ++i) {
                sum += input_data[b * input_size + i] * weights_data[i * output_size + j];
            }
            output_data[b * output_size + j] = sum;
        }
    }

    return output;
}

Tensor DenseLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 2 || grad_output.shape()[1] != output_size) {
        throw std::invalid_argument("Gradient must be 2D with shape {batch_size, output_size}");
    }
    size_t batch_size = grad_output.shape()[0];
    Tensor grad_input({batch_size, input_size});
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    float* weights_data = weights.data();
    float* biases_data = biases.data();
    const float* input_data = input_cache.data();

    // Gradients for input
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < input_size; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < output_size; ++j) {
                sum += grad_output_data[b * output_size + j] * weights_data[i * output_size + j];
            }
            grad_input_data[b * input_size + i] = sum;
        }
    }

    // Update weights and biases
    float lr = learning_rate / batch_size; // Normalize by batch size
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            float grad_w = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                grad_w += input_data[b * input_size + i] * grad_output_data[b * output_size + j];
            }
            weights_data[i * output_size + j] -= lr * grad_w;
        }
    }
    for (size_t j = 0; j < output_size; ++j) {
        float grad_b = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            grad_b += grad_output_data[b * output_size + j];
        }
        biases_data[j] -= lr * grad_b;
    }

    return grad_input;
}

void DenseLayer::setWeights(const Tensor& w) {
    if (w.shape() != weights.shape()) throw std::invalid_argument("Weights shape mismatch");
    weights = w;
}

void DenseLayer::setBiases(const Tensor& b) {
    if (b.shape() != biases.shape()) throw std::invalid_argument("Biases shape mismatch");
    biases = b;
}

void DenseLayer::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
    file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

std::unique_ptr<DenseLayer> DenseLayer::load(std::ifstream& file) {
    size_t input_size, output_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
    auto layer = std::make_unique<DenseLayer>(input_size, output_size);
    file.read(reinterpret_cast<char*>(layer->weights.data()), layer->weights.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->biases.data()), layer->biases.size() * sizeof(float));
    return layer;
}

void DenseLayer::print() const {
    std::cout << "DenseLayer: input_size=" << input_size << ", output_size=" << output_size << "\n";
    std::cout << "Weights:\n";
    weights.print();
    std::cout << "Biases:\n";
    biases.print();
}
