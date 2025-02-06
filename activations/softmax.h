#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor.h"
#include "layers/layer.h" // Наследование от Layer

class Softmax : public Layer {
public:
    Softmax();

    // Прямой проход: применяет Softmax к входным данным
    Tensor forward(const Tensor& input) override;

    // Обратный проход: вычисляет градиент
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    Tensor output_cache; // Кэш выходных данных для использования в backward pass
};

#endif // SOFTMAX_H
