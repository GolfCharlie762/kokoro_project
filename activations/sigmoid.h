#ifndef SIGMOID_H
#define SIGMOID_H

#include "tensor.h"
#include "layers/layer.h" // Наследование от Layer

class Sigmoid : public Layer {
public:
    Sigmoid();

    // Прямой проход: применяет сигмоиду к входным данным
    Tensor forward(const Tensor& input) override;

    // Обратный проход: вычисляет градиент
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    Tensor output_cache; // Кэш выходных данных для использования в backward pass
};

#endif // SIGMOID_H
