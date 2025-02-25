#include "sigmoid.h"
#include <cmath>
#include <fstream>

Sigmoid::Sigmoid() : output_cache() {} // Explicitly use default constructor

Tensor Sigmoid::forward(const Tensor& input) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, features}");
    }
    Tensor output(input.shape());
    float* output_data = output.data();
    const float* input_data = input.data();
    size_t size = input.size();

    for (size_t i = 0; i < size; ++i) {
        output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }

    output_cache = output;
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != output_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match output cache shape");
    }
    Tensor grad_input(output_cache.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* output_data = output_cache.data();
    size_t size = output_cache.size();

    for (size_t i = 0; i < size; ++i) {
        float out = output_data[i];
        grad_input_data[i] = grad_output_data[i] * out * (1.0f - out);
    }

    return grad_input;
}

void Sigmoid::save(std::ofstream& file) const {
    size_t shape_size = output_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(output_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(output_cache.data()), output_cache.size() * sizeof(float));
}

std::unique_ptr<Sigmoid> Sigmoid::load(std::ifstream& file) {
    auto layer = std::make_unique<Sigmoid>();
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->output_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->output_cache.data()), layer->output_cache.size() * sizeof(float));
    return layer;
}

void Sigmoid::print() const {
    std::cout << "Sigmoid Layer\n";
    std::cout << "Output Cache:\n";
    output_cache.print();
}
