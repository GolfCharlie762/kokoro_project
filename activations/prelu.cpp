#include "prelu.h"

// Конструктор с заданным значением alpha по умолчанию
PReLU::PReLU(float alpha) : alpha(alpha), input_cache({}) {}

// Прямой проход
Tensor PReLU::forward(const Tensor& input) {
    // Кэшируем входные данные для использования в backward pass
    input_cache = input;

    // Создаем тензор для выходных данных
    Tensor output(input.shape());

    // Применяем PReLU к каждому элементу входного тензора
    for (size_t i = 0; i < input.size(); ++i) {
        output({i}) = (input({i}) > 0) ? input({i}) : alpha * input({i});
    }

    return output;
}

// Обратный проход
Tensor PReLU::backward(const Tensor& grad_output, float learning_rate) {
    // Создаем тензор для градиента по входным данным
    Tensor grad_input(input_cache.shape());

    // Вычисляем градиент: grad_input = grad_output * (input > 0 ? 1 : alpha)
    for (size_t i = 0; i < input_cache.size(); ++i) {
        grad_input({i}) = grad_output({i}) * ((input_cache({i}) > 0) ? 1.0f : alpha);
    }

    return grad_input;
}
