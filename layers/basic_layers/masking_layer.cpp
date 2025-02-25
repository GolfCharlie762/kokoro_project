#include "layers/basic_layers/masking_layer.h"
#include <fstream>
#include <stdexcept>

MaskingLayer::MaskingLayer(const std::vector<size_t>& input_shape, float mask_value)
    : input_shape(input_shape), mask_value(mask_value), mask(Tensor()), input_cache(Tensor()) {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaskingLayer input shape must be 3D {batch_size, sequence_length, features}");
    }
}

Tensor MaskingLayer::forward(const Tensor& input) {
    if (input.shape().size() != 3 || input.shape()[0] != input_shape[0] ||
        input.shape()[1] != input_shape[1] || input.shape()[2] != input_shape[2]) {
        throw std::invalid_argument("Input shape does not match expected shape in MaskingLayer");
    }
    input_cache = input;
    mask = Tensor(input_shape);
    float* mask_data = mask.data();
    const float* input_data = input.data();
    size_t batch_size = input_shape[0];
    size_t seq_length = input_shape[1];
    size_t features = input_shape[2];

    for (size_t i = 0; i < input.size(); ++i) {
        mask_data[i] = (input_data[i] == mask_value) ? 0.0f : 1.0f;
    }

    Tensor output(input_shape);
    float* output_data = output.data();
    for (size_t i = 0; i < input.size(); ++i) {
        output_data[i] = input_data[i] * mask_data[i];
    }

    return output;
}

Tensor MaskingLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_shape) {
        throw std::invalid_argument("Gradient shape must match input shape in MaskingLayer");
    }
    Tensor grad_input(input_shape);
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* mask_data = mask.data();

    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input_data[i] = grad_output_data[i] * mask_data[i];
    }

    return grad_input;
}

void MaskingLayer::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&mask_value), sizeof(mask_value));
    size_t mask_size = mask.size();
    file.write(reinterpret_cast<const char*>(&mask_size), sizeof(mask_size));
    file.write(reinterpret_cast<const char*>(mask.data()), mask_size * sizeof(float));
    size_t input_cache_size = input_cache.size();
    file.write(reinterpret_cast<const char*>(&input_cache_size), sizeof(input_cache_size));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache_size * sizeof(float));
}

std::unique_ptr<MaskingLayer> MaskingLayer::load(std::ifstream& file) {
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> input_shape(shape_size);
    file.read(reinterpret_cast<char*>(input_shape.data()), shape_size * sizeof(size_t));
    float mask_value;
    file.read(reinterpret_cast<char*>(&mask_value), sizeof(mask_value));
    auto layer = std::make_unique<MaskingLayer>(input_shape, mask_value);

    size_t mask_size;
    file.read(reinterpret_cast<char*>(&mask_size), sizeof(mask_size));
    if (mask_size > 0) {
        layer->mask = Tensor(input_shape);
        file.read(reinterpret_cast<char*>(layer->mask.data()), mask_size * sizeof(float));
    }

    size_t input_cache_size;
    file.read(reinterpret_cast<char*>(&input_cache_size), sizeof(input_cache_size));
    if (input_cache_size > 0) {
        layer->input_cache = Tensor(input_shape);
        file.read(reinterpret_cast<char*>(layer->input_cache.data()), input_cache_size * sizeof(float));
    }

    return layer;
}

void MaskingLayer::print() const {
    std::cout << "MaskingLayer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
    std::cout << "Mask Value: " << mask_value << "\n";
    std::cout << "Mask:\n";
    mask.print();
}
