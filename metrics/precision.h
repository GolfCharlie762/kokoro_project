#ifndef PRECISION_H
#define PRECISION_H

#include "math/tensor.h"

class Precision {
public:
    Precision() = default;
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    float calculate_multiclass(const Tensor& predictions, const Tensor& targets) const;
};

#endif // PRECISION_H
