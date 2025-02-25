#ifndef ACCURACY_H
#define ACCURACY_H

#include "math/tensor.h"

class Accuracy {
public:
    Accuracy() = default;
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    float calculate_argmax(const Tensor& predictions, const Tensor& targets) const; // For classification with logits
};

#endif // ACCURACY_H
