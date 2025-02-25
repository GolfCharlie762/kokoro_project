#include "nadam.h"
#include <cmath>

Nadam::Nadam(float learning_rate, float beta1, float beta2, float epsilon)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Nadam::update(Tensor& param, const Tensor& grad) {
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
        float m_term = (beta1 * m_hat + (1 - beta1) * grad_data[i] / beta1_t);
        param_data[i] -= learning_rate * m_term / (std::sqrt(v_hat) + epsilon);
    }
}

void Nadam::reset() {
    m.clear();
    v.clear();
    t = 0;
}
