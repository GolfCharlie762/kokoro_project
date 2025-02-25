#include "mse.h"
#include <stdexcept>

float MSE::calculate(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = pred_data[i] - targ_data[i];
        sum += diff * diff;
    }
    return sum / predictions.size();
}

Tensor MSE::gradient(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    Tensor grad(predictions.shape());
    float* grad_data = grad.data();
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    float scale = 2.0f / predictions.size();
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad_data[i] = (pred_data[i] - targ_data[i]) * scale;
    }
    return grad;
}
