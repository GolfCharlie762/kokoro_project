#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "math/tensor.h"

class CrossEntropy {
public:
    CrossEntropy() = default;
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    Tensor gradient(const Tensor& predictions, const Tensor& targets) const;
    float calculate_softmax(const Tensor& predictions, const Tensor& targets) const; // For softmax + CE
    Tensor gradient_softmax(const Tensor& predictions, const Tensor& targets) const; // For softmax + CE gradient
};

#endif // CROSS_ENTROPY_H
