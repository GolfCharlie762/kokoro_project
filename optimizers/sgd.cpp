#include "sgd.h"

// Конструктор
SGD::SGD(float learning_rate) : Optimizer(learning_rate) {}

// Обновление параметров
void SGD::update(Tensor& param, const Tensor& grad) {
    for (size_t i = 0; i < param.size(); ++i) {
        param({i}) -= learning_rate * grad({i});
    }
}
