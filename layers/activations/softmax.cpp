#include "softmax.h"
#include <cmath>
#include <fstream>
#include <algorithm>

Softmax::Softmax() : output_cache(Tensor()) {}

Tensor Softmax::forward(const Tensor& input) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, features}");
    }
    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];
    Tensor output({batch_size, features});
    float* output_data = output.data();
    const float* input_data = input.data();

    for (size_t b = 0; b < batch_size; ++b) {
        float max_val = input_data[b * features];
        for (size_t i = 1; i < features; ++i) {
            max_val = std::max(max_val, input_data[b * features + i]);
        }

        float sum_exp = 0.0f;
        for (size_t i = 0; i < features; ++i) {
            output_data[b * features + i] = std::exp(input_data[b * features + i] - max_val);
            sum_exp += output_data[b * features + i];
        }

        for (size_t i = 0; i < features; ++i) {
            output_data[b * features + i] /= sum_exp;
        }
    }

    output_cache = output;
    return output;
}

Tensor Softmax::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != output_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match output cache shape");
    }
    size_t batch_size = output_cache.shape()[0];
    size_t features = output_cache.shape()[1];
    Tensor grad_input({batch_size, features});
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* output_data = output_cache.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < features; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < features; ++j) {
                float out_i = output_data[b * features + i];
                float out_j = output_data[b * features + j];
                float delta_ij = (i == j) ? 1.0f : 0.0f;
                sum += grad_output_data[b * features + j] * out_i * (delta_ij - out_j);
            }
            grad_input_data[b * features + i] = sum;
        }
    }

    return grad_input;
}

void Softmax::save(std::ofstream& file) const {
    size_t shape_size = output_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(output_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(output_cache.data()), output_cache.size() * sizeof(float));
}

std::unique_ptr<Softmax> Softmax::load(std::ifstream& file) {
    auto layer = std::make_unique<Softmax>();
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->output_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->output_cache.data()), layer->output_cache.size() * sizeof(float));
    return layer;
}

void Softmax::print() const {
    std::cout << "Softmax Layer\n";
    std::cout << "Output Cache:\n";
    output_cache.print();
}
