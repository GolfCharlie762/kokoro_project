#include "adamw.h"
#include <cmath>

// Конструктор
AdamW::AdamW(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0) {}

// Обновление параметров
void AdamW::update(Tensor& param, const Tensor& grad) {
    // Инициализация m и v при первом вызове
    if (m.empty()) {
        m.push_back(Tensor(param.shape()));
        v.push_back(Tensor(param.shape()));
    }

    t += 1;

    for (size_t i = 0; i < param.size(); ++i) {
        m[0]({i}) = beta1 * m[0]({i}) + (1 - beta1) * grad({i});
        v[0]({i}) = beta2 * v[0]({i}) + (1 - beta2) * grad({i}) * grad({i});

        float m_hat = m[0]({i}) / (1 - std::pow(beta1, t));
        float v_hat = v[0]({i}) / (1 - std::pow(beta2, t));

        param({i}) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

        // Применение L2-регуляризации
        param({i}) -= learning_rate * weight_decay * param({i});
    }
}
