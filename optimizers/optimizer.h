#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <memory>

class Optimizer {
public:
    // Конструктор
    Optimizer(float learning_rate);

    // Виртуальный деструктор
    virtual ~Optimizer() = default;

    // Обновление параметров
    virtual void update(Tensor& param, const Tensor& grad) = 0;

    float learning_rate; // Скорость обучения
};

#endif // OPTIMIZER_H
