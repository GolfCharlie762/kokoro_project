#include "cross_entropy.h"
#include <stdexcept>
#include <cmath>

float CrossEntropy::calculate(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    size_t size = predictions.size();
    float loss = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        float p = std::max(pred_data[i], 1e-15f); // Numerical stability
        float t = targ_data[i];
        loss += t * std::log(p);
    }

    return -loss / size;
}

Tensor CrossEntropy::gradient(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    Tensor grad(predictions.shape());
    float* grad_data = grad.data();
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    size_t size = predictions.size();

    for (size_t i = 0; i < size; ++i) {
        float p = std::max(pred_data[i], 1e-15f);
        grad_data[i] = -targ_data[i] / p / size;
    }

    return grad;
}

float CrossEntropy::calculate_softmax(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape().size() != 2 || targets.shape().size() != 1 ||
        predictions.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Predictions must be 2D {samples, classes}, targets must be 1D {samples}");
    }
    size_t samples = predictions.shape()[0];
    size_t classes = predictions.shape()[1];
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    float loss = 0.0f;

    for (size_t i = 0; i < samples; ++i) {
        float max_val = pred_data[i * classes];
        for (size_t j = 1; j < classes; ++j) {
            max_val = std::max(max_val, pred_data[i * classes + j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < classes; ++j) {
            sum_exp += std::exp(pred_data[i * classes + j] - max_val);
        }

        size_t target_idx = static_cast<size_t>(targ_data[i]);
        float p = std::exp(pred_data[i * classes + target_idx] - max_val) / sum_exp;
        loss += std::log(std::max(p, 1e-15f));
    }

    return -loss / samples;
}

Tensor CrossEntropy::gradient_softmax(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape().size() != 2 || targets.shape().size() != 1 ||
        predictions.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Predictions must be 2D {samples, classes}, targets must be 1D {samples}");
    }
    size_t samples = predictions.shape()[0];
    size_t classes = predictions.shape()[1];
    Tensor grad(predictions.shape());
    float* grad_data = grad.data();
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();

    for (size_t i = 0; i < samples; ++i) {
        float max_val = pred_data[i * classes];
        for (size_t j = 1; j < classes; ++j) {
            max_val = std::max(max_val, pred_data[i * classes + j]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < classes; ++j) {
            sum_exp += std::exp(pred_data[i * classes + j] - max_val);
        }

        size_t target_idx = static_cast<size_t>(targ_data[i]);
        for (size_t j = 0; j < classes; ++j) {
            float p = std::exp(pred_data[i * classes + j] - max_val) / sum_exp;
            grad_data[i * classes + j] = (p - (j == target_idx ? 1.0f : 0.0f)) / samples;
        }
    }

    return grad;
}
