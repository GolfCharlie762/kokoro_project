#include "layers/basic_layers/identity_layer.h"
#include <fstream>
#include <stdexcept>

IdentityLayer::IdentityLayer(const std::vector<size_t>& input_shape)
    : input_shape(input_shape) {
    if (input_shape.empty()) {
        throw std::invalid_argument("IdentityLayer shape cannot be empty");
    }
}

Tensor IdentityLayer::forward(const Tensor& input) {
    if (input.shape() != input_shape) {
        throw std::invalid_argument("Input shape does not match expected shape in IdentityLayer");
    }
    return input; // Pass through unchanged
}

Tensor IdentityLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_shape) {
        throw std::invalid_argument("Gradient shape must match input shape in IdentityLayer");
    }
    return grad_output; // Pass gradient through unchanged
}

void IdentityLayer::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
}

std::unique_ptr<IdentityLayer> IdentityLayer::load(std::ifstream& file) {
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> input_shape(shape_size);
    file.read(reinterpret_cast<char*>(input_shape.data()), shape_size * sizeof(size_t));
    return std::make_unique<IdentityLayer>(input_shape);
}

void IdentityLayer::print() const {
    std::cout << "IdentityLayer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
}
