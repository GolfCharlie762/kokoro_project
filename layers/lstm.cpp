#include "lstm.h"
#include <cmath>
#include <stdexcept>

// Конструктор
LSTM::LSTM(size_t input_size, size_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      Wf({input_size + hidden_size, hidden_size}), Wi({input_size + hidden_size, hidden_size}),
      Wo({input_size + hidden_size, hidden_size}), Wc({input_size + hidden_size, hidden_size}),
      bf({hidden_size}), bi({hidden_size}), bo({hidden_size}), bc({hidden_size}),
      h_prev({hidden_size}), c_prev({hidden_size}), input_cache({}) {
    // Инициализируем веса случайными значениями
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (size_t i = 0; i < Wf.size(); ++i) Wf({i}) = dist(gen);
    for (size_t i = 0; i < Wi.size(); ++i) Wi({i}) = dist(gen);
    for (size_t i = 0; i < Wo.size(); ++i) Wo({i}) = dist(gen);
    for (size_t i = 0; i < Wc.size(); ++i) Wc({i}) = dist(gen);

    // Инициализируем смещения нулями
    bf.fill(0.0f);
    bi.fill(0.0f);
    bo.fill(0.0f);
    bc.fill(0.0f);

    // Инициализируем предыдущие состояния нулями
    h_prev.fill(0.0f);
    c_prev.fill(0.0f);
}

// Прямой проход
Tensor LSTM::forward(const Tensor& input) {
    // Проверка формы входных данных
    if (input.shape().size() != 1 || input.shape()[0] != input_size) {
        throw std::invalid_argument("Input tensor must have shape (input_size).");
    }

    // Кэшируем входные данные для backward pass
    input_cache = input;

    // Объединяем входные данные и предыдущее скрытое состояние
    Tensor combined({input_size + hidden_size});
    for (size_t i = 0; i < input_size; ++i) combined({i}) = input({i});
    for (size_t i = 0; i < hidden_size; ++i) combined({input_size + i}) = h_prev({i});

    // Вычисляем значения гейтов
    Tensor ft = sigmoid(combined.dot(Wf) + bf); // Forget gate
    Tensor it = sigmoid(combined.dot(Wi) + bi); // Input gate
    Tensor ot = sigmoid(combined.dot(Wo) + bo); // Output gate
    Tensor ct = tanh(combined.dot(Wc) + bc);    // Cell state candidate

    // Обновляем состояние ячейки
    Tensor c_next = ft * c_prev + it * ct;

    // Обновляем скрытое состояние
    Tensor h_next = ot * tanh(c_next);

    // Обновляем предыдущие состояния
    c_prev = c_next;
    h_prev = h_next;

    return h_next;
}

// Обратный проход
Tensor LSTM::backward(const Tensor& grad_output, float learning_rate) {
    // Проверка формы градиента
    if (grad_output.shape().size() != 1 || grad_output.shape()[0] != hidden_size) {
        throw std::invalid_argument("Gradient tensor must have shape (hidden_size).");
    }

    // Градиенты для параметров
    Tensor grad_Wf(Wf.shape());
    Tensor grad_Wi(Wi.shape());
    Tensor grad_Wo(Wo.shape());
    Tensor grad_Wc(Wc.shape());

    Tensor grad_bf(bf.shape());
    Tensor grad_bi(bi.shape());
    Tensor grad_bo(bo.shape());
    Tensor grad_bc(bc.shape());

    // Градиент по скрытому состоянию
    Tensor grad_h_prev(h_prev.shape());
    grad_h_prev.fill(0.0f);

    // Градиент по состоянию ячейки
    Tensor grad_c_prev(c_prev.shape());
    grad_c_prev.fill(0.0f);

    // Объединяем входные данные и предыдущее скрытое состояние
    Tensor combined({input_size + hidden_size});
    for (size_t i = 0; i < input_size; ++i) combined({i}) = input_cache({i});
    for (size_t i = 0; i < hidden_size; ++i) combined({input_size + i}) = h_prev({i});

    // Вычисляем значения гейтов (повторно, как в forward pass)
    Tensor ft = sigmoid(combined.dot(Wf) + bf); // Forget gate
    Tensor it = sigmoid(combined.dot(Wi) + bi); // Input gate
    Tensor ot = sigmoid(combined.dot(Wo) + bo); // Output gate
    Tensor ct = tanh(combined.dot(Wc) + bc);    // Cell state candidate

    // Градиент по выходу (переданный из следующего слоя)
    Tensor grad_h_next = grad_output;

    // Градиент по состоянию ячейки
    Tensor grad_c_next = grad_h_next * ot * (1 - tanh(c_prev) * tanh(c_prev));
    // Градиент по output gate
    Tensor grad_ot = grad_h_next * tanh(c_prev) * ot * (1 - ot);

    // Градиент по cell state candidate
    Tensor grad_ct = grad_c_next * it * (1 - ct * ct);

    // Градиент по input gate
    Tensor grad_it = grad_c_next * ct * it * (1 - it);

    // Градиент по forget gate
    Tensor grad_ft = grad_c_next * c_prev * ft * (1 - ft);

    // Градиент по combined input
    Tensor grad_combined = grad_ft.dot(Wf.transpose()) +
                           grad_it.dot(Wi.transpose()) +
                           grad_ot.dot(Wo.transpose()) +
                           grad_ct.dot(Wc.transpose());

    // Градиент по входным данным
    Tensor grad_input({input_size});
    for (size_t i = 0; i < input_size; ++i) {
        grad_input({i}) = grad_combined({i});
    }

    // Градиент по предыдущему скрытому состоянию
    for (size_t i = 0; i < hidden_size; ++i) {
        grad_h_prev({i}) = grad_combined({input_size + i});
    }

    // Градиент по предыдущему состоянию ячейки
    grad_c_prev = grad_c_next * ft;

    // Обновляем параметры
    for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < input_size + hidden_size; ++j) {
            grad_Wf({j, i}) += grad_ft({i}) * combined({j});
            grad_Wi({j, i}) += grad_it({i}) * combined({j});
            grad_Wo({j, i}) += grad_ot({i}) * combined({j});
            grad_Wc({j, i}) += grad_ct({i}) * combined({j});
        }
        grad_bf({i}) += grad_ft({i});
        grad_bi({i}) += grad_it({i});
        grad_bo({i}) += grad_ot({i});
        grad_bc({i}) += grad_ct({i});
    }

    // Обновляем веса и смещения
    for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < input_size + hidden_size; ++j) {
            Wf({j, i}) -= learning_rate * grad_Wf({j, i});
            Wi({j, i}) -= learning_rate * grad_Wi({j, i});
            Wo({j, i}) -= learning_rate * grad_Wo({j, i});
            Wc({j, i}) -= learning_rate * grad_Wc({j, i});
        }
        bf({i}) -= learning_rate * grad_bf({i});
        bi({i}) -= learning_rate * grad_bi({i});
        bo({i}) -= learning_rate * grad_bo({i});
        bc({i}) -= learning_rate * grad_bc({i});
    }

    return grad_input;
}

// Вспомогательная функция: сигмоида
Tensor LSTM::sigmoid(const Tensor& x) const {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result({i}) = 1.0f / (1.0f + std::exp(-x({i})));
    }
    return result;
}

// Вспомогательная функция: гиперболический тангенс
Tensor LSTM::tanh(const Tensor& x) const {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result({i}) = std::tanh(x({i}));
    }
    return result;
}

