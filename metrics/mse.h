#ifndef MSE_H
#define MSE_H

#include "math/tensor.h"

class MSE {
public:
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    Tensor gradient(const Tensor& predictions, const Tensor& targets) const;
};

#endif // MSE_H
