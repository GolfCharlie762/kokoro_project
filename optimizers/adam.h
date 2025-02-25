#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"
#include <vector>

class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void update(Tensor& param, const Tensor& grad) override;
    void reset(); // Reset momentum and velocity

private:
    float beta1, beta2, epsilon;
    std::vector<Tensor> m; // Momentum
    std::vector<Tensor> v; // Velocity
    size_t t; // Timestep
};

#endif // ADAM_H
