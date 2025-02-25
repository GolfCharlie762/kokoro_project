#include "flatten.h"
#include <fstream>

Flatten::Flatten() : input_shape() {}

Tensor Flatten::forward(const Tensor& input) {
    if (input.shape().size() < 2) {
        throw std::invalid_argument("Input must have at least 2 dimensions for Flatten");
    }
    input_shape = input.shape();
    size_t batch_size = input_shape[0];
    size_t features = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        features *= input_shape[i];
    }
    return input.reshape({batch_size, features});
}

Tensor Flatten::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 2 || grad_output.shape()[0] != input_shape[0]) {
        throw std::invalid_argument("Gradient shape must match flattened output shape");
    }
    return grad_output.reshape(input_shape);
}

void Flatten::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
}

std::unique_ptr<Flatten> Flatten::load(std::ifstream& file) {
    auto layer = std::make_unique<Flatten>();
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    layer->input_shape.resize(shape_size);
    file.read(reinterpret_cast<char*>(layer->input_shape.data()), shape_size * sizeof(size_t));
    return layer;
}

void Flatten::print() const {
    std::cout << "Flatten Layer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
}
