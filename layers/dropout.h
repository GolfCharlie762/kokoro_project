#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"
#include "layer.h"
#include <random>

class Dropout : public Layer {
public:
    Dropout(float rate);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    float rate; // Вероятность отключения нейронов
    Tensor mask; // Маска для отключения нейронов
    std::mt19937 gen; // Генератор случайных чисел
    std::uniform_real_distribution<float> dist; // Распределение для генерации случайных чисел
};

#endif // DROPOUT_H
