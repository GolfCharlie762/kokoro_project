#include "gelu.h"
#include <fstream>
#include <cmath>

GELU::GELU() : input_cache(Tensor()) {}

Tensor GELU::forward(const Tensor& input) {
    input_cache = input;
    Tensor output(input.shape());
    float* output_data = output.data();
    const float* input_data = input.data();
    size_t size = input.size();

    for (size_t i = 0; i < size; ++i) {
        output_data[i] = gelu(input_data[i]);
    }

    return output;
}

Tensor GELU::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match input shape in GELU");
    }
    Tensor grad_input(input_cache.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* input_data = input_cache.data();
    size_t size = grad_output.size();

    for (size_t i = 0; i < size; ++i) {
        grad_input_data[i] = grad_output_data[i] * gelu_derivative(input_data[i]);
    }

    return grad_input;
}

void GELU::save(std::ofstream& file) const {
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<GELU> GELU::load(std::ifstream& file) {
    auto layer = std::make_unique<GELU>();
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));
    return layer;
}

void GELU::print() const {
    std::cout << "GELU Layer\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
}

float GELU::gelu(float x) const {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    return x * 0.5f * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
}

float GELU::gelu_derivative(float x) const {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    float tanh_arg = sqrt_2_pi * (x + 0.044715f * x * x * x);
    float tanh_val = std::tanh(tanh_arg);
    float sech_sq = 1.0f - tanh_val * tanh_val;
    return 0.5f * (1.0f + tanh_val) + x * 0.5f * sqrt_2_pi * sech_sq * (1.0f + 3.0f * 0.044715f * x * x);
}
