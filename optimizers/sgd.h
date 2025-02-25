#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

class SGD : public Optimizer {
public:
    SGD(float learning_rate, float momentum = 0.0f);
    void update(Tensor& param, const Tensor& grad) override;
    void reset();

private:
    float momentum;
    std::vector<Tensor> velocity; // For momentum
};

#endif // SGD_H
