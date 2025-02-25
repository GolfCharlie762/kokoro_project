#include "sgd.h"

SGD::SGD(float learning_rate, float momentum)
    : Optimizer(learning_rate), momentum(momentum) {}

void SGD::update(Tensor& param, const Tensor& grad) {
    if (param.shape() != grad.shape()) {
        throw std::invalid_argument("Parameter and gradient shapes must match");
    }
    if (velocity.empty()) {
        velocity.emplace_back(param.shape(), 0.0f);
    }

    float* param_data = param.data();
    const float* grad_data = grad.data();
    float* vel_data = velocity[0].data();
    size_t size = param.size();

    for (size_t i = 0; i < size; ++i) {
        vel_data[i] = momentum * vel_data[i] - learning_rate * grad_data[i];
        param_data[i] += vel_data[i];
    }
}

void SGD::reset() {
    velocity.clear();
}
