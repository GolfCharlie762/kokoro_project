#include "elu.h"
#include <cmath>
#include <fstream>

ELU::ELU(float alpha) : input_cache(Tensor()), alpha(alpha) {}

Tensor ELU::forward(const Tensor& input) {
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
        output_data[i] = (x > 0) ? x : alpha * (std::exp(x) - 1);
    }

    return output;
}

Tensor ELU::backward(const Tensor& grad_output, float learning_rate) {
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
        grad_input_data[i] = grad_output_data[i] * ((x > 0) ? 1.0f : alpha * std::exp(x));
    }

    return grad_input;
}

void ELU::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<ELU> ELU::load(std::ifstream& file) {
    float alpha;
    file.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
    auto layer = std::make_unique<ELU>(alpha);
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));
    return layer;
}

void ELU::print() const {
    std::cout << "ELU Layer (alpha=" << alpha << ")\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
}
