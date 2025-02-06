#include "sigmoid.h"
#include <cmath>

// Конструктор по умолчанию
Sigmoid::Sigmoid() : output_cache({}) {} // Инициализируем output_cache с пустой формой

// Прямой проход
Tensor Sigmoid::forward(const Tensor& input) {
    // Создаем тензор для выходных данных
    Tensor output(input.shape());

    // Применяем сигмоиду к каждому элементу входного тензора
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input({i});
        output({i}) = 1.0f / (1.0f + std::exp(-x));
    }

    // Кэшируем выходные данные для использования в backward pass
    output_cache = output;

    return output;
}

// Обратный проход
Tensor Sigmoid::backward(const Tensor& grad_output, float learning_rate) {
    // Создаем тензор для градиента по входным данным
    Tensor grad_input(output_cache.shape());

    // Вычисляем градиент: grad_input = grad_output * (output_cache * (1 - output_cache))
    for (size_t i = 0; i < output_cache.size(); ++i) {
        float output_val = output_cache({i});
        grad_input({i}) = grad_output({i}) * output_val * (1.0f - output_val);
    }

    return grad_input;
}
