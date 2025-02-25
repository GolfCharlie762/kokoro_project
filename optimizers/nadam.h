#ifndef NADAM_H
#define NADAM_H

#include "optimizer.h"
#include <vector>

class Nadam : public Optimizer {
public:
    Nadam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void update(Tensor& param, const Tensor& grad) override;
    void reset();

private:
    float beta1, beta2, epsilon;
    std::vector<Tensor> m; // Momentum
    std::vector<Tensor> v; // Velocity
    size_t t; // Timestep
};

#endif // NADAM_H
