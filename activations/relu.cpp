#include "relu.h"

// Конструктор по умолчанию
ReLU::ReLU() : input_cache({}) {} // Инициализируем input_cache с пустой формой

// Прямой проход
Tensor ReLU::forward(const Tensor& input) {
    // Кэшируем входные данные для использования в backward pass
    input_cache = input;

    // Создаем тензор для выходных данных
    Tensor output(input.shape());

    // Применяем ReLU к каждому элементу входного тензора
    for (size_t i = 0; i < input.size(); ++i) {
        output({i}) = std::max(0.0f, input({i}));
    }

    return output;
}

// Обратный проход
Tensor ReLU::backward(const Tensor& grad_output, float learning_rate) {
    // Создаем тензор для градиента по входным данным
    Tensor grad_input(input_cache.shape());

    // Вычисляем градиент: grad_input = grad_output * (input > 0 ? 1 : 0)
    for (size_t i = 0; i < input_cache.size(); ++i) {
        grad_input({i}) = grad_output({i}) * (input_cache({i}) > 0 ? 1.0f : 0.0f);
    }

    return grad_input;
}
