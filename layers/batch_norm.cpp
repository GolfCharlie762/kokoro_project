#include "batch_norm.h"
#include <cmath>

// Конструктор
BatchNorm::BatchNorm(size_t num_features, float epsilon, float momentum)
    : num_features(num_features), epsilon(epsilon), momentum(momentum),
      gamma({num_features}), beta({num_features}),
      running_mean({num_features}), running_var({num_features}),
      input_cache({}), normalized({}) {
    // Инициализируем параметры
    gamma.fill(1.0f); // Начальное значение gamma = 1
    beta.fill(0.0f);  // Начальное значение beta = 0

    // Инициализируем скользящие средние
    running_mean.fill(0.0f);
    running_var.fill(1.0f);
}

// Прямой проход
Tensor BatchNorm::forward(const Tensor& input) {
    // Проверка формы входных данных
    if (input.shape().size() != 2 || input.shape()[1] != num_features) {
        throw std::invalid_argument("Input tensor must have shape (batch_size, num_features).");
    }

    // Кэшируем входные данные для backward pass
    input_cache = input;

    // Вычисляем среднее значение и дисперсию по мини-батчу
    Tensor mean({num_features});
    Tensor var({num_features});

    for (size_t i = 0; i < num_features; ++i) {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (size_t j = 0; j < input.shape()[0]; ++j) {
            sum += input({j, i});
            sum_sq += input({j, i}) * input({j, i});
        }

        mean({i}) = sum / input.shape()[0];
        var({i}) = (sum_sq / input.shape()[0]) - (mean({i}) * mean({i}));
    }

    // Обновляем скользящие средние
    for (size_t i = 0; i < num_features; ++i) {
        running_mean({i}) = momentum * running_mean({i}) + (1 - momentum) * mean({i});
        running_var({i}) = momentum * running_var({i}) + (1 - momentum) * var({i});
    }

    // Нормализуем данные
    normalized = Tensor(input.shape());
    for (size_t i = 0; i < input.shape()[0]; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            normalized({i, j}) = (input({i, j}) - mean({j})) / std::sqrt(var({j}) + epsilon);
        }
    }

    // Применяем масштабирование и смещение
    Tensor output(input.shape());
    for (size_t i = 0; i < input.shape()[0]; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            output({i, j}) = gamma({j}) * normalized({i, j}) + beta({j});
        }
    }

    return output;
}

// Обратный проход
Tensor BatchNorm::backward(const Tensor& grad_output, float learning_rate) {
    // Проверка формы градиента
    if (grad_output.shape() != input_cache.shape()) {
        throw std::invalid_argument("Gradient tensor must have the same shape as input tensor.");
    }

    // Вычисляем градиенты для gamma и beta
    Tensor grad_gamma({num_features});
    Tensor grad_beta({num_features});

    for (size_t j = 0; j < num_features; ++j) {
        float sum_gamma = 0.0f;
        float sum_beta = 0.0f;

        for (size_t i = 0; i < grad_output.shape()[0]; ++i) {
            sum_gamma += grad_output({i, j}) * normalized({i, j});
            sum_beta += grad_output({i, j});
        }

        grad_gamma({j}) = sum_gamma;
        grad_beta({j}) = sum_beta;
    }

    // Обновляем параметры gamma и beta
    for (size_t j = 0; j < num_features; ++j) {
        gamma({j}) -= learning_rate * grad_gamma({j});
        beta({j}) -= learning_rate * grad_beta({j});
    }

    // Вычисляем градиент по входным данным
    Tensor grad_input(input_cache.shape());
    for (size_t i = 0; i < grad_input.shape()[0]; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            grad_input({i, j}) = grad_output({i, j}) * gamma({j}) / std::sqrt(running_var({j}) + epsilon);
        }
    }

    return grad_input;
}
