#ifndef ADAMAX_H
#define ADAMAX_H

#include "optimizer.h"

class Adamax : public Optimizer {
public:
    Adamax(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    // Обновление параметров
    void update(Tensor& param, const Tensor& grad) override;

private:
    float beta1, beta2, epsilon;
    std::vector<Tensor> m; // Скользящее среднее градиента
    std::vector<Tensor> u; // Максимум абсолютных значений градиента
    size_t t; // Счетчик шагов
};

#endif // ADAMAX_H
