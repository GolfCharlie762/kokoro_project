#include "f1_score.h"
#include <stdexcept>
#include <vector>

float F1Score::calculate(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();
    size_t size = predictions.size();
    size_t true_positives = 0;
    size_t false_positives = 0;
    size_t false_negatives = 0;

    for (size_t i = 0; i < size; ++i) {
        bool pred = pred_data[i] > 0.5f; // Assuming binary classification with threshold 0.5
        bool targ = targ_data[i] > 0.5f;
        if (pred && targ) true_positives++;
        else if (pred && !targ) false_positives++;
        else if (!pred && targ) false_negatives++;
    }

    float precision = static_cast<float>(true_positives) / (true_positives + false_positives + 1e-15f);
    float recall = static_cast<float>(true_positives) / (true_positives + false_negatives + 1e-15f);
    return 2.0f * (precision * recall) / (precision + recall + 1e-15f);
}

float F1Score::calculate_multiclass(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.shape().size() != 2 || targets.shape().size() != 1 ||
        predictions.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Predictions must be 2D {samples, classes}, targets must be 1D {samples}");
    }
    size_t samples = predictions.shape()[0];
    size_t classes = predictions.shape()[1];
    const float* pred_data = predictions.data();
    const float* targ_data = targets.data();

    std::vector<size_t> true_positives(classes, 0);
    std::vector<size_t> false_positives(classes, 0);
    std::vector<size_t> false_negatives(classes, 0);

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
        size_t targ_idx = static_cast<size_t>(targ_data[i]);
        if (pred_max_idx == targ_idx) {
            true_positives[targ_idx]++;
        } else {
            false_positives[pred_max_idx]++;
            false_negatives[targ_idx]++;
        }
    }

    float macro_f1 = 0.0f;
    for (size_t c = 0; c < classes; ++c) {
        float precision = static_cast<float>(true_positives[c]) / (true_positives[c] + false_positives[c] + 1e-15f);
        float recall = static_cast<float>(true_positives[c]) / (true_positives[c] + false_negatives[c] + 1e-15f);
        macro_f1 += 2.0f * (precision * recall) / (precision + recall + 1e-15f);
    }

    return macro_f1 / classes;
}
