#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "tensor.h"
#include "layer.h"
#include <cmath>

class BatchNorm : public Layer {
public:
    BatchNorm(size_t num_features, float epsilon = 1e-5, float momentum = 0.9);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    size_t num_features; // Количество признаков (каналов)
    float epsilon;       // Малое значение для численной стабильности
    float momentum;      // Коэффициент для скользящего среднего

    Tensor gamma;        // Параметр масштабирования
    Tensor beta;         // Параметр смещения

    Tensor running_mean; // Скользящее среднее для среднего значения
    Tensor running_var;  // Скользящее среднее для дисперсии

    Tensor input_cache;  // Кэш входных данных для backward pass
    Tensor normalized;   // Нормализованные данные
};

#endif // BATCH_NORM_H
