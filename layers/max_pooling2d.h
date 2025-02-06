#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include "tensor.h"
#include "layers/layer.h"

class MaxPooling2D : public Layer {
public:
    MaxPooling2D(size_t pool_size, size_t stride = 2);

    // Прямой проход
    Tensor forward(const Tensor& input) override;

    // Обратный проход
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

private:
    size_t pool_size, stride;
    Tensor input_cache; // Кэш входных данных для использования в backward pass
};

#endif // MAX_POOLING2D_H
