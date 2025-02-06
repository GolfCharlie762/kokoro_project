#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>

// Базовый класс для всех слоев
class Layer {
public:
    virtual ~Layer() = default;

    // Прямой проход: вычисляет выходные данные на основе входных
    virtual Tensor forward(const Tensor& input) = 0;

    // Обратный проход: вычисляет градиенты и обновляет параметры слоя
    virtual Tensor backward(const Tensor& grad_output, float learning_rate) = 0;
};

#endif // LAYER_H
