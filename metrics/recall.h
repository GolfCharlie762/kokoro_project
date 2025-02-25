#ifndef RECALL_H
#define RECALL_H

#include "math/tensor.h"

class Recall {
public:
    Recall() = default;
    float calculate(const Tensor& predictions, const Tensor& targets) const;
    float calculate_multiclass(const Tensor& predictions, const Tensor& targets) const;
};

#endif // RECALL_H
