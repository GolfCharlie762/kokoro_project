#include "accuracy.h"
#include <stdexcept>
#include <algorithm>

float Accuracy::calculate(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    size_t size = predictions.size();
    size_t correct = 0;

    for (size_t i = 0; i < size; ++i) {
        if (pred_data[i] == targ_data[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / size;
}

float Accuracy::calculate_argmax(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape().size() != 2 || targets.shape().size() != 1 ||
        predictions.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Predictions must be 2D {samples, classes}, targets must be 1D {samples}");
    }
    size_t samples = predictions.shape()[0];
    size_t classes = predictions.shape()[1];
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    size_t correct = 0;

    for (size_t i = 0; i < samples; ++i) {
        size_t pred_idx = i * classes;
        size_t pred_max_idx = 0;
        float max_val = pred_data[pred_idx];
        for (size_t j = 1; j < classes; ++j) {
            if (pred_data[pred_idx + j] > max_val) {
                max_val = pred_data[pred_idx + j];
                pred_max_idx = j;
            }
        }
        if (static_cast<size_t>(targ_data[i]) == pred_max_idx) {
            correct++;
        }
    }

    return static_cast<float>(correct) / samples;
}
