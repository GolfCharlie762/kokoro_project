#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "tensor.h"

// Полносвязный слой
class DenseLayer : public Layer {
public:
    // Конструктор
    DenseLayer(size_t input_size, size_t output_size);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    // Получить веса и смещения (для отладки)
    Tensor getWeights() const;
    Tensor getBiases() const;

private:
    size_t input_size;  // Размер входных данных
    size_t output_size; // Размер выходных данных
    Tensor weights;     // Матрица весов (input_size x output_size)
    Tensor biases;      // Вектор смещений (output_size)
    Tensor input_cache; // Кэш входных данных для использования в backward pass
};

#endif // DENSE_LAYER_H
