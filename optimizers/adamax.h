#ifndef ADAMAX_H
#define ADAMAX_H

#include "optimizer.h"
#include <vector>

class Adamax : public Optimizer {
public:
    Adamax(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void update(Tensor& param, const Tensor& grad) override;
    void reset(); // Reset momentum and max gradient

private:
    float beta1, beta2, epsilon;
    std::vector<Tensor> m; // Momentum
    std::vector<Tensor> u; // Max absolute gradient
    size_t t; // Timestep
};

#endif // ADAMAX_H
