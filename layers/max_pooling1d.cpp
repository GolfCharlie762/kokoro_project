#include "max_pooling1d.h"
#include <fstream>
#include <algorithm>

MaxPooling1D::MaxPooling1D(size_t pool_size, size_t stride)
    : pool_size(pool_size), stride(stride), input_cache(Tensor()), max_indices(Tensor()) {}

Tensor MaxPooling1D::forward(const Tensor& input) {
    if (input.shape().size() != 3) {
        throw std::invalid_argument("Input must be 3D with shape {batch_size, channels, length}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    size_t channels = input.shape()[1];
    size_t input_length = input.shape()[2];
    size_t output_length = (input_length - pool_size) / stride + 1;

    Tensor output({batch_size, channels, output_length});
    max_indices = Tensor({batch_size, channels, output_length});
    float* output_data = output.data();
    float* max_indices_data = max_indices.data();
    const float* input_data = input.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t ol = 0; ol < output_length; ++ol) {
                size_t out_idx = b * channels * output_length + c * output_length + ol;
                float max_val = -std::numeric_limits<float>::infinity();
                size_t max_idx = 0;

                for (size_t pl = 0; pl < pool_size; ++pl) {
                    size_t il = ol * stride + pl;
                    if (il < input_length) {
                        size_t in_idx = b * channels * input_length + c * input_length + il;
                        float val = input_data[in_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = in_idx;
                        }
                    }
                }
                output_data[out_idx] = max_val;
                max_indices_data[out_idx] = static_cast<float>(max_idx);
            }
        }
    }

    return output;
}

Tensor MaxPooling1D::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 3 || grad_output.shape()[2] != (input_cache.shape()[2] - pool_size) / stride + 1) {
        throw std::invalid_argument("Gradient shape must match output shape from forward pass");
    }
    Tensor grad_input(input_cache.shape());
    grad_input.fill(0.0f);
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* max_indices_data = max_indices.data();
    size_t size = grad_output.size();

    for (size_t i = 0; i < size; ++i) {
        size_t max_idx = static_cast<size_t>(max_indices_data[i]);
        grad_input_data[max_idx] += grad_output_data[i];
    }

    return grad_input;
}

void MaxPooling1D::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&pool_size), sizeof(pool_size));
    file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<MaxPooling1D> MaxPooling1D::load(std::ifstream& file) {
    size_t pool_size, stride;
    file.read(reinterpret_cast<char*>(&pool_size), sizeof(pool_size));
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    auto layer = std::make_unique<MaxPooling1D>(pool_size, stride);
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    layer->max_indices = Tensor(shape); // Assuming same shape for simplicity; adjust if needed
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));
    return layer;
}

void MaxPooling1D::print() const {
    std::cout << "MaxPooling1D Layer: pool_size=" << pool_size << ", stride=" << stride << "\n";
    std::cout << "Input Cache:\n";
    input_cache.print();
    std::cout << "Max Indices:\n";
    max_indices.print();
}
