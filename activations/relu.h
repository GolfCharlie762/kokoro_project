#ifndef RELU_H
#define RELU_H

#include "tensor.h"
#include "layers/layer.h" // Добавляем наследование от Layer

class ReLU : public Layer {
public:
    ReLU();

    // Реализация методов интерфейса Layer
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    Tensor input_cache;
};

#endif // RELU_H
