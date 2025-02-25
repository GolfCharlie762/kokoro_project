#include "model.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// Optimizer interface
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Tensor& weights, const Tensor& grad, float learning_rate) = 0;
};

class SGD : public Optimizer {
public:
    void update(Tensor& weights, const Tensor& grad, float learning_rate) override {
        weights = weights - grad * Tensor({1}, learning_rate); // Element-wise scaling
    }
};

// Model implementation
Model::Model(LossType loss_type)
    : loss_type_(loss_type), optimizer_(std::make_unique<SGD>()) {}

void Model::addLayer(std::shared_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("Cannot add null layer to model");
    }
    layers.push_back(layer);
}

Tensor Model::predict(const Tensor& input) {
    if (input.shape().empty()) {
        throw std::runtime_error("Input tensor is empty in predict");
    }
    Tensor output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Model::train(const Tensor& input, const Tensor& target, size_t epochs,
                  float learning_rate, size_t batch_size) {
    if (input.shape().empty() || target.shape().empty()) {
        throw std::runtime_error("Input or target tensor is empty in train");
    }
    size_t num_samples = input.shape()[0]; // Assuming input shape is [samples, ...]
    if (num_samples != target.shape()[0]) {
        throw std::runtime_error("Mismatch in number of samples between input and target");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t num_batches = 0;

        for (size_t start = 0; start < num_samples; start += batch_size) {
            size_t end = std::min(start + batch_size, num_samples);
            Tensor batch_input = input.slice(start, end);   // Extract batch
            Tensor batch_target = target.slice(start, end); // Extract corresponding targets

            // Forward pass
            Tensor output = predict(batch_input);

            // Compute loss
            float batch_loss = compute_loss(output, batch_target);
            total_loss += batch_loss;
            num_batches++;

            // Compute gradient for backpropagation
            Tensor grad_output;
            if (loss_type_ == LossType::MSE) {
                grad_output = output - batch_target; // MSE gradient
            } else { // CrossEntropy
                grad_output = output - batch_target; // Simplified; assumes softmax in last layer
                // For strict cross-entropy with softmax, gradient is handled in layer
            }

            // Backward pass
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad_output = (*it)->backward(grad_output, learning_rate);
            }
        }

        float avg_loss = total_loss / num_batches;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << avg_loss << std::endl;
    }
}

float Model::validate(const Tensor& input, const Tensor& target) {
    if (input.shape().empty() || target.shape().empty()) {
        throw std::runtime_error("Input or target tensor is empty in validation");
    }
    if (input.shape()[0] != target.shape()[0]) {
        throw std::runtime_error("Mismatch in number of samples between input and target in validation");
    }
    Tensor output = predict(input);
    return compute_loss(output, target);
}

const std::vector<std::shared_ptr<Layer>>& Model::getLayers() const {
    return layers;
}

void Model::setOptimizer(std::unique_ptr<Optimizer> optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("Optimizer cannot be null");
    }
    optimizer_ = std::move(optimizer);
}

float Model::compute_loss(const Tensor& output, const Tensor& target) const {
    if (output.shape() != target.shape()) {
        throw std::runtime_error("Output and target shapes do not match in loss computation");
    }
    if (loss_type_ == LossType::MSE) {
        Tensor diff = output - target;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < diff.size(); ++i) {
            sum_sq += diff.at(i) * diff.at({i});
        }
        return sum_sq / diff.size(); // MSE
    } else { // CrossEntropy
        float loss = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            loss -= target.at(i) * std::log(output.at(i) + 1e-10f); // Avoid log(0)
        }
        return loss / output.shape()[0]; // Average over samples
    }
}
