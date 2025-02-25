#include "adamw.h"
#include <cmath>

AdamW::AdamW(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0) {}

void AdamW::update(Tensor& param, const Tensor& grad) {
    if (param.shape() != grad.shape()) {
        throw std::invalid_argument("Parameter and gradient shapes must match");
    }
    if (m.empty()) {
        m.emplace_back(param.shape(), 0.0f);
        v.emplace_back(param.shape(), 0.0f);
    }

    t++;
    float* param_data = param.data();
    const float* grad_data = grad.data();
    float* m_data = m[0].data();
    float* v_data = v[0].data();
    size_t size = param.size();
    float beta1_t = 1.0f - std::pow(beta1, t);
    float beta2_t = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < size; ++i) {
        m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
        v_data[i] = beta2 * v_data[i] + (1 - beta2) * grad_data[i] * grad_data[i];
        float m_hat = m_data[i] / beta1_t;
        float v_hat = v_data[i] / beta2_t;
        param_data[i] -= learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon) + weight_decay * param_data[i]);
    }
}

void AdamW::reset() {
    m.clear();
    v.clear();
    t = 0;
}
