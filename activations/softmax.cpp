#include "softmax.h"
#include <cmath>
#include <stdexcept>

// Конструктор по умолчанию
Softmax::Softmax() : output_cache({}) {} // Инициализируем output_cache с пустой формой

// Прямой проход
Tensor Softmax::forward(const Tensor& input) {
    // Создаем тензор для выходных данных
    Tensor output(input.shape());

    // Вычисляем экспоненты и их сумму
    float max_val = input({0});
    for (size_t i = 1; i < input.size(); ++i) {
        if (input({i}) > max_val) {
            max_val = input({i});
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output({i}) = std::exp(input({i}) - max_val); // Для численной стабильности
        sum_exp += output({i});
    }

    // Нормализуем выходные данные
    for (size_t i = 0; i < input.size(); ++i) {
        output({i}) /= sum_exp;
    }

    // Кэшируем выходные данные для использования в backward pass
    output_cache = output;

    return output;
}

// Обратный проход
Tensor Softmax::backward(const Tensor& grad_output, float learning_rate) {
    // Создаем тензор для градиента по входным данным
    Tensor grad_input(output_cache.shape());

    // Вычисляем градиент: grad_input = grad_output * (output_cache * (1 - output_cache))
    for (size_t i = 0; i < output_cache.size(); ++i) {
        float output_val = output_cache({i});
        grad_input({i}) = grad_output({i}) * output_val * (1.0f - output_val);
    }

    return grad_input;
}
