#ifndef PRELU_H
#define PRELU_H

#include "math/tensor.h"
#include "layers/layer.h"

class PReLU : public Layer {
public:
    PReLU(float alpha = 0.25f); // Конструктор с параметром alpha

    // Реализация методов интерфейса Layer
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    float alpha; // Параметр наклона для отрицательных значений
    Tensor input_cache; // Кэшируем входные данные для обратного прохода
};

#endif // PRELU_H
