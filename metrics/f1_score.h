#ifndef F1_SCORE_H
#define F1_SCORE_H

#include "math/tensor.h"

class F1Score {
public:
    F1Score() = default;
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    float calculate_multiclass(const Tensor& predictions, const Tensor& targets) const; // For multi-class with argmax
};

#endif // F1_SCORE_H
