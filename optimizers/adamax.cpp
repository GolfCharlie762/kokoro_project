#include "adamax.h"
#include <cmath>

Adamax::Adamax(float learning_rate, float beta1, float beta2, float epsilon)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adamax::update(Tensor& param, const Tensor& grad) {
    if (param.shape() != grad.shape()) {
        throw std::invalid_argument("Parameter and gradient shapes must match");
    }
    if (m.empty()) {
        m.emplace_back(param.shape(), 0.0f);
        u.emplace_back(param.shape(), 0.0f);
    }

    t++;
    float* param_data = param.data();
    const float* grad_data = grad.data();
    float* m_data = m[0].data();
    float* u_data = u[0].data();
    size_t size = param.size();
    float beta1_t = 1.0f - std::pow(beta1, t);

    for (size_t i = 0; i < size; ++i) {
        m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
        u_data[i] = std::max(beta2 * u_data[i], std::abs(grad_data[i]));
        param_data[i] -= learning_rate * (m_data[i] / beta1_t) / (u_data[i] + epsilon);
    }
}

void Adamax::reset() {
    m.clear();
    u.clear();
    t = 0;
}
