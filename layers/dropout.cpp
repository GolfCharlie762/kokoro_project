#include "dropout.h"

// Конструктор
Dropout::Dropout(float rate)
    : rate(rate), mask({}), gen(std::random_device{}()), dist(0.0f, 1.0f) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in the range [0, 1).");
    }
}

// Прямой проход
Tensor Dropout::forward(const Tensor& input) {
    // Генерируем маску для отключения нейронов
    mask = Tensor(input.shape());
    for (size_t i = 0; i < mask.size(); ++i) {
        mask({i}) = (dist(gen) < rate) ? 0.0f : 1.0f / (1.0f - rate);
    }

    // Применяем маску к входным данным
    Tensor output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        output({i}) = input({i}) * mask({i});
    }

    return output;
}

// Обратный проход
Tensor Dropout::backward(const Tensor& grad_output, float learning_rate) {
    // Применяем маску к градиенту
    Tensor grad_input(grad_output.shape());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input({i}) = grad_output({i}) * mask({i});
    }

    return grad_input;
}
