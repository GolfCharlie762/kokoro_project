#include "lstm.h"
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>

LSTM::LSTM(size_t input_size, size_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      Wf({input_size + hidden_size, hidden_size}),
      Wi({input_size + hidden_size, hidden_size}),
      Wo({input_size + hidden_size, hidden_size}),
      Wc({input_size + hidden_size, hidden_size}),
      bf({hidden_size}), bi({hidden_size}),
      bo({hidden_size}), bc({hidden_size}),
      h_prev({hidden_size}), c_prev({hidden_size}),
      input_cache(Tensor()) {
    float limit = std::sqrt(6.0f / (input_size + hidden_size + hidden_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    std::generate(Wf.data(), Wf.data() + Wf.size(), [&]() { return dis(gen); });
    std::generate(Wi.data(), Wi.data() + Wi.size(), [&]() { return dis(gen); });
    std::generate(Wo.data(), Wo.data() + Wo.size(), [&]() { return dis(gen); });
    std::generate(Wc.data(), Wc.data() + Wc.size(), [&]() { return dis(gen); });

    bf.fill(0.0f);
    bi.fill(0.0f);
    bo.fill(0.0f);
    bc.fill(0.0f);
    h_prev.fill(0.0f);
    c_prev.fill(0.0f);
}

Tensor LSTM::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != input_size) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, input_size}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    Tensor h_next({batch_size, hidden_size});
    Tensor c_next({batch_size, hidden_size});
    float* h_next_data = h_next.data();
    float* c_next_data = c_next.data();
    const float* input_data = input.data();

    for (size_t b = 0; b < batch_size; ++b) {
        Tensor combined({input_size + hidden_size});
        float* combined_data = combined.data();
        const float* h_prev_data = h_prev.data();
        for (size_t i = 0; i < input_size; ++i) {
            combined_data[i] = input_data[b * input_size + i];
        }
        for (size_t i = 0; i < hidden_size; ++i) {
            combined_data[input_size + i] = h_prev_data[i];
        }

        Tensor ft = sigmoid(combined.dot(Wf) + bf);
        Tensor it = sigmoid(combined.dot(Wi) + bi);
        Tensor ot = sigmoid(combined.dot(Wo) + bo);
        Tensor ct = tanh(combined.dot(Wc) + bc);

        const float* ft_data = ft.data();
        const float* it_data = it.data();
        const float* ot_data = ot.data();
        const float* ct_data = ct.data();
        const float* c_prev_data = c_prev.data();

        for (size_t i = 0; i < hidden_size; ++i) {
            c_next_data[b * hidden_size + i] = ft_data[i] * c_prev_data[i] + it_data[i] * ct_data[i];
            h_next_data[b * hidden_size + i] = ot_data[i] * std::tanh(c_next_data[b * hidden_size + i]);
        }
    }

    h_prev = h_next.slice(batch_size - 1, batch_size); // Update with last time step
    c_prev = c_next.slice(batch_size - 1, batch_size);
    return h_next;
}

Tensor LSTM::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 2 || grad_output.shape()[1] != hidden_size) {
        throw std::invalid_argument("Gradient must be 2D with shape {batch_size, hidden_size}");
    }
    size_t batch_size = grad_output.shape()[0];
    Tensor grad_input({batch_size, input_size});
    Tensor grad_h_prev({1, hidden_size});
    Tensor grad_c_prev({1, hidden_size});
    grad_h_prev.fill(0.0f);
    grad_c_prev.fill(0.0f);

    float* grad_input_data = grad_input.data();
    float* grad_h_prev_data = grad_h_prev.data();
    float* grad_c_prev_data = grad_c_prev.data();
    float* wf_data = Wf.data();
    float* wi_data = Wi.data();
    float* wo_data = Wo.data();
    float* wc_data = Wc.data();
    float* bf_data = bf.data();
    float* bi_data = bi.data();
    float* bo_data = bo.data();
    float* bc_data = bc.data();
    const float* input_data = input_cache.data();
    const float* grad_output_data = grad_output.data();

    for (int b = batch_size - 1; b >= 0; --b) {
        Tensor combined({input_size + hidden_size});
        float* combined_data = combined.data();
        const float* h_prev_data = h_prev.data();
        for (size_t i = 0; i < input_size; ++i) {
            combined_data[i] = input_data[b * input_size + i];
        }
        for (size_t i = 0; i < hidden_size; ++i) {
            combined_data[input_size + i] = h_prev_data[i];
        }

        Tensor ft = sigmoid(combined.dot(Wf) + bf);
        Tensor it = sigmoid(combined.dot(Wi) + bi);
        Tensor ot = sigmoid(combined.dot(Wo) + bo);
        Tensor ct = tanh(combined.dot(Wc) + bc);

        Tensor grad_h_next({hidden_size});
        float* grad_h_next_data = grad_h_next.data();
        for (size_t i = 0; i < hidden_size; ++i) {
            grad_h_next_data[i] = grad_output_data[b * hidden_size + i] + grad_h_prev_data[i];
        }

        const float* ft_data = ft.data();
        const float* it_data = it.data();
        const float* ot_data = ot.data();
        const float* ct_data = ct.data();
        const float* c_prev_data = c_prev.data();

        Tensor grad_c_next({hidden_size});
        float* grad_c_next_data = grad_c_next.data();
        for (size_t i = 0; i < hidden_size; ++i) {
            grad_c_next_data[i] = grad_h_next_data[i] * ot_data[i] * (1 - std::tanh(c_prev_data[i]) * std::tanh(c_prev_data[i]));
        }

        Tensor grad_ot({hidden_size});
        Tensor grad_ct({hidden_size});
        Tensor grad_it({hidden_size});
        Tensor grad_ft({hidden_size});
        float* grad_ot_data = grad_ot.data();
        float* grad_ct_data = grad_ct.data();
        float* grad_it_data = grad_it.data();
        float* grad_ft_data = grad_ft.data();

        for (size_t i = 0; i < hidden_size; ++i) {
            grad_ot_data[i] = grad_h_next_data[i] * std::tanh(c_prev_data[i]) * ot_data[i] * (1 - ot_data[i]);
            grad_ct_data[i] = grad_c_next_data[i] * it_data[i] * (1 - ct_data[i] * ct_data[i]);
            grad_it_data[i] = grad_c_next_data[i] * ct_data[i] * it_data[i] * (1 - it_data[i]);
            grad_ft_data[i] = grad_c_next_data[i] * c_prev_data[i] * ft_data[i] * (1 - ft_data[i]);
        }

        Tensor grad_combined = grad_ft.dot(Wf.transpose()) + grad_it.dot(Wi.transpose()) +
                               grad_ot.dot(Wo.transpose()) + grad_ct.dot(Wc.transpose());

        const float* grad_combined_data = grad_combined.data();
        for (size_t i = 0; i < input_size; ++i) {
            grad_input_data[b * input_size + i] = grad_combined_data[i];
        }
        for (size_t i = 0; i < hidden_size; ++i) {
            grad_h_prev_data[i] = grad_combined_data[input_size + i];
        }

        float lr = learning_rate / batch_size;
        for (size_t j = 0; j < input_size + hidden_size; ++j) {
            for (size_t i = 0; i < hidden_size; ++i) {
                wf_data[j * hidden_size + i] -= lr * grad_ft_data[i] * combined_data[j];
                wi_data[j * hidden_size + i] -= lr * grad_it_data[i] * combined_data[j];
                wo_data[j * hidden_size + i] -= lr * grad_ot_data[i] * combined_data[j];
                wc_data[j * hidden_size + i] -= lr * grad_ct_data[i] * combined_data[j];
            }
        }
        for (size_t i = 0; i < hidden_size; ++i) {
            bf_data[i] -= lr * grad_ft_data[i];
            bi_data[i] -= lr * grad_it_data[i];
            bo_data[i] -= lr * grad_ot_data[i];
            bc_data[i] -= lr * grad_ct_data[i];
        }

        grad_c_prev = grad_c_next * ft;
        c_prev = c_prev.slice(0, 1); // Reset for next iteration
        h_prev = h_prev.slice(0, 1);
    }

    return grad_input;
}

Tensor LSTM::sigmoid(const Tensor& x) const {
    Tensor result(x.shape());
    float* result_data = result.data();
    const float* x_data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-x_data[i]));
    }
    return result;
}

Tensor LSTM::tanh(const Tensor& x) const {
    Tensor result(x.shape());
    float* result_data = result.data();
    const float* x_data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        result_data[i] = std::tanh(x_data[i]);
    }
    return result;
}

void LSTM::setWeights(const Tensor& wf, const Tensor& wi, const Tensor& wo, const Tensor& wc,
                      const Tensor& bf, const Tensor& bi, const Tensor& bo, const Tensor& bc) {
    if (wf.shape() != Wf.shape() || wi.shape() != Wi.shape() || wo.shape() != Wo.shape() || wc.shape() != Wc.shape() ||
        bf.shape() != this->bf.shape() || bi.shape() != this->bi.shape() ||
        bo.shape() != this->bo.shape() || bc.shape() != this->bc.shape()) {
        throw std::invalid_argument("Shape mismatch in LSTM weights or biases");
    }
    Wf = wf;
    Wi = wi;
    Wo = wo;
    Wc = wc;
    this->bf = bf;
    this->bi = bi;
    this->bo = bo;
    this->bc = bc;
}

void LSTM::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    file.write(reinterpret_cast<const char*>(Wf.data()), Wf.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Wi.data()), Wi.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Wo.data()), Wo.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Wc.data()), Wc.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bf.data()), bf.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bi.data()), bi.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bo.data()), bo.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bc.data()), bc.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_prev.data()), h_prev.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(c_prev.data()), c_prev.size() * sizeof(float));
}

std::unique_ptr<LSTM> LSTM::load(std::ifstream& file) {
    size_t input_size, hidden_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    auto layer = std::make_unique<LSTM>(input_size, hidden_size);
    file.read(reinterpret_cast<char*>(layer->Wf.data()), layer->Wf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Wi.data()), layer->Wi.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Wo.data()), layer->Wo.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Wc.data()), layer->Wc.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bf.data()), layer->bf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bi.data()), layer->bi.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bo.data()), layer->bo.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bc.data()), layer->bc.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->h_prev.data()), layer->h_prev.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->c_prev.data()), layer->c_prev.size() * sizeof(float));
    return layer;
}

void LSTM::print() const {
    std::cout << "LSTM Layer: input_size=" << input_size << ", hidden_size=" << hidden_size << "\n";
    std::cout << "Wf:\n"; Wf.print();
    std::cout << "Wi:\n"; Wi.print();
    std::cout << "Wo:\n"; Wo.print();
    std::cout << "Wc:\n"; Wc.print();
    std::cout << "bf:\n"; bf.print();
    std::cout << "bi:\n"; bi.print();
    std::cout << "bo:\n"; bo.print();
    std::cout << "bc:\n"; bc.print();
    std::cout << "h_prev:\n"; h_prev.print();
    std::cout << "c_prev:\n"; c_prev.print();
}
