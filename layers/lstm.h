#ifndef LSTM_H
#define LSTM_H

#include "tensor.h"
#include "layer.h"
#include <vector>
#include <random>

class LSTM : public Layer {
public:
    LSTM(size_t input_size, size_t hidden_size);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    size_t input_size;  // Размер входных данных
    size_t hidden_size; // Размер скрытого состояния

    // Параметры LSTM
    Tensor Wf, Wi, Wo, Wc; // Веса для forget, input, output и cell gate
    Tensor bf, bi, bo, bc; // Смещения

    // Кэши для обратного прохода
    Tensor input_cache; // Кэш входных данных
    Tensor h_prev;      // Предыдущее скрытое состояние
    Tensor c_prev;      // Предыдущее состояние ячейки

    // Вспомогательные функции
    Tensor sigmoid(const Tensor& x) const;
    Tensor tanh(const Tensor& x) const;
};

#endif // LSTM_H
