#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

class SGD : public Optimizer {
public:
    SGD(float learning_rate);

    // Обновление параметров
    void update(Tensor& param, const Tensor& grad) override;
};

#endif // SGD_H
