#ifndef ADAMW_H
#define ADAMW_H

#include "optimizer.h"

class AdamW : public Optimizer {
public:
    AdamW(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f);

    // Обновление параметров
    void update(Tensor& param, const Tensor& grad) override;

private:
    float beta1, beta2, epsilon;
    float weight_decay;
    std::vector<Tensor> m; // Скользящее среднее градиента
    std::vector<Tensor> v; // Скользящее среднее квадрата градиента
    size_t t; // Счетчик шагов
};

#endif // ADAMW_H
