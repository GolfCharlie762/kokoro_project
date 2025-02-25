#include "swish.h"
#include <cmath>
#include <fstream>

Swish::Swish() : input_cache(Tensor()) {}

Tensor Swish::forward(const Tensor& input) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, features}");
    }
    input_cache = input;
    Tensor output(input.shape());
    float* output_data = output.data();
    const float* input_data = input.data();
    size_t size = input.size();

    for (size_t i = 0; i < size; ++i) {
        float x = input_data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        output_data[i] = x * sigmoid;
    }

    return output;
}

Tensor Swish::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match input cache shape");
    }
    Tensor grad_input(input_cache.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* input_data = input_cache.data();
    size_t size = input_cache.size();

    for (size_t i = 0; i < size; ++i) {
        float x = input_data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        float swish = x * sigmoid;
        grad_input_data[i] = grad_output_data[i] * (swish + sigmoid * (1.0f - swish));
    }

    return grad_input;
}

void Swish::save(std::ofstream& file) const {
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<Swish> Swish::load(std::ifstream& file) {
    auto layer = std::make_unique<Swish>();
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));
    return layer;
}

void Swish::print() const {
    std::cout << "Swish Layer\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
}
