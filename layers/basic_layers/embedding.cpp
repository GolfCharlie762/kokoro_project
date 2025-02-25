#include "embedding.h"
#include <random>
#include <fstream>
#include <cmath>

Embedding::Embedding(size_t vocab_size, size_t embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim),
      weights({vocab_size, embedding_dim}), input_cache(Tensor()) {
    float limit = std::sqrt(6.0f / (vocab_size + embedding_dim));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);
    float* weights_data = weights.data();
    for (size_t i = 0; i < weights.size(); ++i) {
        weights_data[i] = dis(gen); // Xavier initialization
    }
}

Tensor Embedding::forward(const Tensor& input) {
    if (input.shape().size() < 2) {
        throw std::invalid_argument("Input must be at least 2D with shape {batch_size, sequence_length, ...}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    size_t seq_length = input.shape()[1];
    std::vector<size_t> output_shape = {batch_size, seq_length, embedding_dim};
    Tensor output(output_shape);
    float* output_data = output.data();
    const float* input_data = input.data();
    const float* weights_data = weights.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            size_t index = static_cast<size_t>(input_data[b * seq_length + t]);
            if (index >= vocab_size) {
                throw std::out_of_range("Embedding index " + std::to_string(index) + " exceeds vocab_size " + std::to_string(vocab_size));
            }
            size_t weight_offset = index * embedding_dim;
            size_t output_offset = (b * seq_length + t) * embedding_dim;
            std::copy(weights_data + weight_offset, weights_data + weight_offset + embedding_dim, output_data + output_offset);
        }
    }

    return output;
}

Tensor Embedding::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 3 || grad_output.shape()[2] != embedding_dim) {
        throw std::invalid_argument("Gradient must be 3D with shape {batch_size, sequence_length, embedding_dim}");
    }
    size_t batch_size = input_cache.shape()[0];
    size_t seq_length = input_cache.shape()[1];
    Tensor grad_input(input_cache.shape());
    grad_input.fill(0.0f); // No gradient flows back to input indices

    const float* grad_output_data = grad_output.data(); // Changed to const float*
    float* weights_data = weights.data();
    const float* input_data = input_cache.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_length; ++t) {
            size_t index = static_cast<size_t>(input_data[b * seq_length + t]);
            size_t weight_offset = index * embedding_dim;
            size_t grad_offset = (b * seq_length + t) * embedding_dim;
            for (size_t d = 0; d < embedding_dim; ++d) {
                weights_data[weight_offset + d] -= learning_rate * grad_output_data[grad_offset + d];
            }
        }
    }

    return grad_input;
}

void Embedding::setWeights(const Tensor& new_weights) {
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("Shape mismatch in embedding weights");
    }
    weights = new_weights;
}

void Embedding::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));
    file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
}

std::unique_ptr<Embedding> Embedding::load(std::ifstream& file) {
    size_t vocab_size, embedding_dim;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(embedding_dim));
    auto layer = std::make_unique<Embedding>(vocab_size, embedding_dim);
    file.read(reinterpret_cast<char*>(layer->weights.data()), layer->weights.size() * sizeof(float));
    return layer;
}

void Embedding::print() const {
    std::cout << "Embedding Layer: vocab_size=" << vocab_size << ", embedding_dim=" << embedding_dim << "\n";
    std::cout << "Weights:\n";
    weights.print();
}
