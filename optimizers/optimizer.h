#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "math/tensor.h"

class Optimizer {
public:
    Optimizer(float learning_rate);
    virtual ~Optimizer() = default;
    virtual void update(Tensor& param, const Tensor& grad) = 0;

    float learning_rate;
};

#endif // OPTIMIZER_H
