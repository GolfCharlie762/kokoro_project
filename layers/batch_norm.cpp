#include "batch_norm.h"
#include <cmath>
#include <fstream>

BatchNorm::BatchNorm(size_t num_features, float epsilon, float momentum)
    : num_features(num_features), epsilon(epsilon), momentum(momentum),
      gamma({num_features}), beta({num_features}),
      running_mean({num_features}), running_var({num_features}),
      input_cache(Tensor()), normalized(Tensor()) {
    gamma.fill(1.0f);
    beta.fill(0.0f);
    running_mean.fill(0.0f);
    running_var.fill(1.0f);
}

Tensor BatchNorm::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != num_features) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, num_features}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    Tensor mean({num_features});
    Tensor var({num_features});
    float* mean_data = mean.data();
    float* var_data = var.data();
    const float* input_data = input.data();

    // Compute mean and variance
    for (size_t j = 0; j < num_features; ++j) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            float val = input_data[i * num_features + j];
            sum += val;
            sum_sq += val * val;
        }
        mean_data[j] = sum / batch_size;
        var_data[j] = (sum_sq / batch_size) - (mean_data[j] * mean_data[j]);
    }

    // Update running statistics
    float* running_mean_data = running_mean.data();
    float* running_var_data = running_var.data();
    for (size_t j = 0; j < num_features; ++j) {
        running_mean_data[j] = momentum * running_mean_data[j] + (1 - momentum) * mean_data[j];
        running_var_data[j] = momentum * running_var_data[j] + (1 - momentum) * var_data[j];
    }

    // Normalize and scale
    normalized = Tensor(input.shape());
    Tensor output(input.shape());
    float* normalized_data = normalized.data();
    float* output_data = output.data();
    const float* gamma_data = gamma.data();
    const float* beta_data = beta.data();

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            size_t idx = i * num_features + j;
            normalized_data[idx] = (input_data[idx] - mean_data[j]) / std::sqrt(var_data[j] + epsilon);
            output_data[idx] = gamma_data[j] * normalized_data[idx] + beta_data[j];
        }
    }

    return output;
}

Tensor BatchNorm::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != input_cache.shape()) {
        throw std::invalid_argument("Gradient shape must match input shape");
    }
    size_t batch_size = input_cache.shape()[0];
    Tensor grad_gamma({num_features});
    Tensor grad_beta({num_features});
    float* grad_gamma_data = grad_gamma.data();
    float* grad_beta_data = grad_beta.data();
    const float* grad_output_data = grad_output.data();
    const float* normalized_data = normalized.data();

    // Compute gradients for gamma and beta
    for (size_t j = 0; j < num_features; ++j) {
        float sum_gamma = 0.0f;
        float sum_beta = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = i * num_features + j;
            sum_gamma += grad_output_data[idx] * normalized_data[idx];
            sum_beta += grad_output_data[idx];
        }
        grad_gamma_data[j] = sum_gamma;
        grad_beta_data[j] = sum_beta;
    }

    // Update gamma and beta
    float* gamma_data = gamma.data();
    float* beta_data = beta.data();
    float lr = learning_rate / batch_size; // Normalize by batch size
    for (size_t j = 0; j < num_features; ++j) {
        gamma_data[j] -= lr * grad_gamma_data[j];
        beta_data[j] -= lr * grad_beta_data[j];
    }

    // Compute gradient for input
    Tensor grad_input(input_cache.shape());
    float* grad_input_data = grad_input.data(); // Corrected: Get pointer from Tensor object
    const float* input_data = input_cache.data();
    const float* running_var_data = running_var.data();

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            size_t idx = i * num_features + j;
            grad_input_data[idx] = grad_output_data[idx] * gamma_data[j] / std::sqrt(running_var_data[j] + epsilon);
        }
    }

    return grad_input;
}

void BatchNorm::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    file.write(reinterpret_cast<const char*>(&momentum), sizeof(momentum));
    file.write(reinterpret_cast<const char*>(gamma.data()), gamma.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(beta.data()), beta.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(running_mean.data()), running_mean.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(running_var.data()), running_var.size() * sizeof(float));
}

std::unique_ptr<BatchNorm> BatchNorm::load(std::ifstream& file) {
    size_t num_features;
    float epsilon, momentum;
    file.read(reinterpret_cast<char*>(&num_features), sizeof(num_features));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    file.read(reinterpret_cast<char*>(&momentum), sizeof(momentum));
    auto layer = std::make_unique<BatchNorm>(num_features, epsilon, momentum);
    file.read(reinterpret_cast<char*>(layer->gamma.data()), layer->gamma.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->beta.data()), layer->beta.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->running_mean.data()), layer->running_mean.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->running_var.data()), layer->running_var.size() * sizeof(float));
    return layer;
}

void BatchNorm::print() const {
    std::cout << "BatchNorm Layer: num_features=" << num_features
              << ", epsilon=" << epsilon << ", momentum=" << momentum << "\n";
    std::cout << "Gamma:\n"; gamma.print();
    std::cout << "Beta:\n"; beta.print();
    std::cout << "Running Mean:\n"; running_mean.print();
    std::cout << "Running Variance:\n"; running_var.print();
}
