#include "layers/basic_layers/lambda_layer.h"
#include <fstream>
#include <stdexcept>

LambdaLayer::LambdaLayer(std::function<float(float)> forward_fn,
                         std::function<float(float)> backward_fn,
                         const std::vector<size_t>& input_shape)
    : forward_fn(forward_fn), backward_fn(backward_fn), input_shape(input_shape), input_cache(Tensor()) {
    if (!forward_fn || !backward_fn) {
        throw std::invalid_argument("LambdaLayer requires valid forward and backward functions");
    }
    if (input_shape.empty()) {
        throw std::invalid_argument("LambdaLayer input shape cannot be empty");
    }
}

Tensor LambdaLayer::forward(const Tensor& input) {
    if (input.shape() != input_shape) {
        throw std::invalid_argument("Input shape does not match expected shape in LambdaLayer");
    }
    input_cache = input;
    Tensor output(input_shape);
    float* output_data = output.data();
    const float* input_data = input.data();

    for (size_t i = 0; i < input.size(); ++i) {
        output_data[i] = forward_fn(input_data[i]);
    }

    return output;
}

Tensor LambdaLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_shape) {
        throw std::invalid_argument("Gradient shape must match input shape in LambdaLayer");
    }
    Tensor grad_input(input_shape);
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* input_data = input_cache.data();

    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input_data[i] = grad_output_data[i] * backward_fn(input_data[i]);
    }

    return grad_input;
}

void LambdaLayer::save(std::ofstream& file) const {
    size_t shape_size = input_shape.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_shape.data()), shape_size * sizeof(size_t));
    // Note: Lambda functions cannot be serialized; assume reconstruction via predefined functions
    size_t cache_size = input_cache.size();
    file.write(reinterpret_cast<const char*>(&cache_size), sizeof(cache_size));
    file.write(reinterpret_cast<const char*>(input_cache.data()), cache_size * sizeof(float));
}

std::unique_ptr<LambdaLayer> LambdaLayer::load(std::ifstream& file) {
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> input_shape(shape_size);
    file.read(reinterpret_cast<char*>(input_shape.data()), shape_size * sizeof(size_t));

    // Default to identity function if no custom function provided; user must reset post-load
    auto default_fn = [](float x) { return x; };
    auto layer = std::make_unique<LambdaLayer>(default_fn, default_fn, input_shape);

    size_t cache_size;
    file.read(reinterpret_cast<char*>(&cache_size), sizeof(cache_size));
    if (cache_size > 0) {
        layer->input_cache = Tensor(input_shape);
        file.read(reinterpret_cast<char*>(layer->input_cache.data()), cache_size * sizeof(float));
    }

    return layer;
}

void LambdaLayer::print() const {
    std::cout << "LambdaLayer\n";
    std::cout << "Input Shape: {";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
}
