#include "layers/basic_layers/input_layer.h"
#include <fstream>
#include <stdexcept>

InputLayer::InputLayer(const std::vector<size_t>& input_shape)
    : input_shape(input_shape) {
    if (input_shape.empty()) {
        throw std::invalid_argument("InputLayer shape cannot be empty");
    }
}

Tensor InputLayer::forward(const Tensor& input) {
    if (input.shape() != input_shape) {
        throw std::invalid_argument("Input shape does not match expected shape");
    }
    return input; // Pass through unchanged
}

Tensor InputLayer::backward(const Tensor& grad_output, float learning_rate) {
    // Input layer has no parameters to update, return zero gradient of same shape as input
    Tensor grad_input(input_shape);
    grad_input.fill(0.0f);
    return grad_input;
}

void InputLayer::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
}

std::unique_ptr<InputLayer> InputLayer::load(std::ifstream& file) {
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> input_shape(shape_size);
    file.read(reinterpret_cast<char*>(input_shape.data()), shape_size * sizeof(size_t));
    return std::make_unique<InputLayer>(input_shape);
}

void InputLayer::print() const {
    std::cout << "InputLayer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
}
