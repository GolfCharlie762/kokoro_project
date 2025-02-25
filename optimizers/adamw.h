#ifndef ADAMW_H
#define ADAMW_H

#include "optimizer.h"
#include <vector>

class AdamW : public Optimizer {
public:
    AdamW(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f);
    void update(Tensor& param, const Tensor& grad) override;
    void reset();

private:
    float beta1, beta2, epsilon, weight_decay;
    std::vector<Tensor> m; // Momentum
    std::vector<Tensor> v; // Velocity
    size_t t; // Timestep
};

#endif // ADAMW_H
