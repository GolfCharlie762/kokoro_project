#include "dense_layer.h"
#include <random>
#include <iostream>

// Конструктор
DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size),
      weights({input_size, output_size}), biases({output_size}), input_cache({input_size}) {
    // Инициализируем веса случайными значениями
    weights.randomize(-0.5f, 0.5f);
    // Инициализируем смещения нулями
    biases.fill(0.0f);
}

// Прямой проход
Tensor DenseLayer::forward(const Tensor& input) {
    // Кэшируем входные данные для использования в backward pass
    input_cache = input;

    // Создаем тензор для выходных данных
    Tensor output({output_size});

    // Вычисляем выходные данные: output = input * weights + biases
    for (size_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; ++j) {
            sum += input({j}) * weights({j, i});
        }
        output({i}) = sum + biases({i});
    }

    return output;
}

// Обратный проход
Tensor DenseLayer::backward(const Tensor& grad_output, float learning_rate) {
    // Вычисляем градиенты для весов и смещений
    Tensor grad_input({input_size});
    Tensor grad_weights(weights.shape());
    Tensor grad_biases(biases.shape());

    // Градиент по входным данным: grad_input = grad_output * weights^T
    for (size_t i = 0; i < input_size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < output_size; ++j) {
            sum += grad_output({j}) * weights({i, j});
        }
        grad_input({i}) = sum;
    }

    // Градиент по весам: grad_weights = input^T * grad_output
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            grad_weights({i, j}) = input_cache({i}) * grad_output({j});
        }
    }

    // Градиент по смещениям: grad_biases = grad_output
    for (size_t i = 0; i < output_size; ++i) {
        grad_biases({i}) = grad_output({i});
    }

    // Обновляем веса и смещения
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weights({i, j}) -= learning_rate * grad_weights({i, j});
        }
    }
    for (size_t i = 0; i < output_size; ++i) {
        biases({i}) -= learning_rate * grad_biases({i});
    }

    return grad_input;
}

// Получить веса (для отладки)
Tensor DenseLayer::getWeights() const {
    return weights;
}

// Получить смещения (для отладки)
Tensor DenseLayer::getBiases() const {
    return biases;
}
