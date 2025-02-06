#include "conv2d.h"
#include <random>
#include <iostream>
#include <stdexcept>

// Конструктор
Conv2D::Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride, size_t padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), stride(stride), padding(padding),
      kernels({output_channels, input_channels, kernel_size, kernel_size}), biases({output_channels}), input_cache({}) {
    // Инициализируем ядра случайными значениями
    kernels.randomize(-0.5f, 0.5f);

    // Инициализируем смещения нулями
    biases.fill(0.0f);
}

// Прямой проход
Tensor Conv2D::forward(const Tensor& input) {
    // Проверка формы входных данных
    if (input.shape().size() != 3 || input.shape()[0] != input_channels) {
        throw std::invalid_argument("Input tensor must have shape (input_channels, height, width).");
    }

    // Кэшируем входные данные для использования в backward pass
    input_cache = input;

    // Применяем padding к входным данным
    Tensor padded_input = pad(input);

    // Вычисляем размеры выходного тензора
    size_t output_height = (padded_input.shape()[1] - kernel_size) / stride + 1;
    size_t output_width = (padded_input.shape()[2] - kernel_size) / stride + 1;
    Tensor output({output_channels, output_height, output_width});

    // Применяем свертку
    for (size_t oc = 0; oc < output_channels; ++oc) { // По каждому выходному каналу
        for (size_t oh = 0; oh < output_height; ++oh) { // По высоте выходного тензора
            for (size_t ow = 0; ow < output_width; ++ow) { // По ширине выходного тензора
                float sum = 0.0f;

                // Применяем ядро свертки
                for (size_t ic = 0; ic < input_channels; ++ic) { // По каждому входному каналу
                    for (size_t kh = 0; kh < kernel_size; ++kh) { // По высоте ядра
                        for (size_t kw = 0; kw < kernel_size; ++kw) { // По ширине ядра
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            sum += padded_input({ic, ih, iw}) * kernels({oc, ic, kh, kw});
                        }
                    }
                }

                // Добавляем смещение
                output({oc, oh, ow}) = sum + biases({oc});
            }
        }
    }

    return output;
}

// Обратный проход
Tensor Conv2D::backward(const Tensor& grad_output, float learning_rate) {
    // Проверка формы градиента
    if (grad_output.shape().size() != 3 || grad_output.shape()[0] != output_channels) {
        throw std::invalid_argument("Gradient tensor must have shape (output_channels, height, width).");
    }

    // Инициализируем градиенты для ядер и смещений
    Tensor grad_kernels(kernels.shape());
    Tensor grad_biases(biases.shape());

    // Вычисляем градиенты для смещений
    for (size_t oc = 0; oc < output_channels; ++oc) {
        float sum = 0.0f;
        for (size_t oh = 0; oh < grad_output.shape()[1]; ++oh) {
            for (size_t ow = 0; ow < grad_output.shape()[2]; ++ow) {
                sum += grad_output({oc, oh, ow});
            }
        }
        grad_biases({oc}) = sum;
    }

    // Вычисляем градиенты для ядер
    Tensor padded_input = pad(input_cache);
    for (size_t oc = 0; oc < output_channels; ++oc) {
        for (size_t ic = 0; ic < input_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    float sum = 0.0f;
                    for (size_t oh = 0; oh < grad_output.shape()[1]; ++oh) {
                        for (size_t ow = 0; ow < grad_output.shape()[2]; ++ow) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            sum += padded_input({ic, ih, iw}) * grad_output({oc, oh, ow});
                        }
                    }
                    grad_kernels({oc, ic, kh, kw}) = sum;
                }
            }
        }
    }

    // Обновляем ядра и смещения
    for (size_t oc = 0; oc < output_channels; ++oc) {
        for (size_t ic = 0; ic < input_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    kernels({oc, ic, kh, kw}) -= learning_rate * grad_kernels({oc, ic, kh, kw});
                }
            }
        }
        biases({oc}) -= learning_rate * grad_biases({oc});
    }

    // Вычисляем градиент по входным данным (обратная свертка)
    Tensor grad_input(input_cache.shape());
    grad_input.fill(0.0f); // Инициализируем нулями

    // Применяем обратную свертку
    for (size_t oc = 0; oc < output_channels; ++oc) {
        for (size_t oh = 0; oh < grad_output.shape()[1]; ++oh) {
            for (size_t ow = 0; ow < grad_output.shape()[2]; ++ow) {
                for (size_t ic = 0; ic < input_channels; ++ic) {
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;

                            // Проверяем, что индексы находятся в пределах границ
                            if (ih < grad_input.shape()[1] && iw < grad_input.shape()[2]) {
                                grad_input({ic, ih, iw}) += kernels({oc, ic, kh, kw}) * grad_output({oc, oh, ow});
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}


// Вспомогательная функция: добавление padding к входным данным
Tensor Conv2D::pad(const Tensor& input) const {
    if (padding == 0) {
        return input;
    }

    // Создаем тензор с padding
    Tensor padded_input({input.shape()[0], input.shape()[1] + 2 * padding, input.shape()[2] + 2 * padding});
    padded_input.fill(0.0f);

    // Копируем данные в центр тензора
    for (size_t c = 0; c < input.shape()[0]; ++c) {
        for (size_t h = 0; h < input.shape()[1]; ++h) {
            for (size_t w = 0; w < input.shape()[2]; ++w) {
                padded_input({c, h + padding, w + padding}) = input({c, h, w});
            }
        }
    }

    return padded_input;
}
