#include "layers/activations/celu.h"
#include <fstream>
#include <cmath>

CELU::CELU(float alpha)
    : alpha(alpha), input_cache(Tensor()) {
    if (alpha <= 0.0f) {
        throw std::invalid_argument("CELU alpha must be positive");
    }
}

Tensor CELU::forward(const Tensor& input) {
    input_cache = input;
    Tensor output(input.shape());
    float* output_data = output.data();
    const float* input_data = input.data();
    size_t size = input.size();

    for (size_t i = 0; i < size; ++i) {
        output_data[i] = celu(input_data[i]);
    }

    return output;
}

Tensor CELU::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match input shape in CELU");
    }
    Tensor grad_input(input_cache.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* input_data = input_cache.data();
    size_t size = grad_output.size();

    for (size_t i = 0; i < size; ++i) {
        grad_input_data[i] = grad_output_data[i] * celu_derivative(input_data[i]);
    }

    return grad_input;
}

float CELU::celu(float x) const {
    return (x >= 0.0f) ? x : alpha * (std::exp(x / alpha) - 1.0f);
}

float CELU::celu_derivative(float x) const {
    return (x >= 0.0f) ? 1.0f : std::exp(x / alpha);
}

void CELU::setAlpha(float new_alpha) {
    if (new_alpha <= 0.0f) {
        throw std::invalid_argument("CELU alpha must be positive");
    }
    alpha = new_alpha;
}

void CELU::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<CELU> CELU::load(std::ifstream& file) {
    float alpha;
    file.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
    auto layer = std::make_unique<CELU>(alpha);
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));
    return layer;
}

void CELU::print() const {
    std::cout << "CELU Layer\n";
    std::cout << "Alpha: " << alpha << "\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
}
