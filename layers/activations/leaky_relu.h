#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "math/tensor.h"
#include "layers/layer.h"

class LeakyReLU : public Layer {
public:
    explicit LeakyReLU(float alpha = 0.01f); // Конструктор с параметром alpha

    // Реализация методов интерфейса Layer
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    float alpha; // Параметр наклона для отрицательных значений
    Tensor input_cache; // Кэшируем входные данные
};

#endif // LEAKY_RELU_H
