#include "adamax.h"
#include <cmath>

// Конструктор
Adamax::Adamax(float learning_rate, float beta1, float beta2, float epsilon)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

// Обновление параметров
void Adamax::update(Tensor& param, const Tensor& grad) {
    // Инициализация m и u при первом вызове
    if (m.empty()) {
        m.push_back(Tensor(param.shape()));
        u.push_back(Tensor(param.shape()));
    }

    t += 1;

    for (size_t i = 0; i < param.size(); ++i) {
        m[0]({i}) = beta1 * m[0]({i}) + (1 - beta1) * grad({i});
        u[0]({i}) = std::max(beta2 * u[0]({i}), std::abs(grad({i})));

        param({i}) -= learning_rate * m[0]({i}) / (u[0]({i}) + epsilon);
    }
}

