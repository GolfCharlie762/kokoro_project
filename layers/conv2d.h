#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"
#include "layer.h"

class Conv2D : public Layer {
public:
    Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    size_t input_channels, output_channels, kernel_size, stride, padding;
    Tensor kernels; // Ядра свертки (фильтры)
    Tensor biases;  // Смещения
    Tensor input_cache; // Кэш входных данных для использования в backward pass

    // Вспомогательные функции
    Tensor pad(const Tensor& input) const;
    Tensor convolve(const Tensor& input, const Tensor& kernel) const;
};

#endif // CONV2D_H
