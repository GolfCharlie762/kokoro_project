#include "gru.h"
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>

GRU::GRU(size_t input_size, size_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      Wz({input_size, hidden_size}), Wr({input_size, hidden_size}), Wh({input_size, hidden_size}),
      Uz({hidden_size, hidden_size}), Ur({hidden_size, hidden_size}), Uh({hidden_size, hidden_size}),
      bz({hidden_size}), br({hidden_size}), bh({hidden_size}),
      h_prev({hidden_size}), input_cache(Tensor()), zt_cache(Tensor()), rt_cache(Tensor()), ht_cache(Tensor()) {
    float limit = std::sqrt(6.0f / (input_size + hidden_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    std::generate(Wz.data(), Wz.data() + Wz.size(), [&]() { return dis(gen); });
    std::generate(Wr.data(), Wr.data() + Wr.size(), [&]() { return dis(gen); });
    std::generate(Wh.data(), Wh.data() + Wh.size(), [&]() { return dis(gen); });
    std::generate(Uz.data(), Uz.data() + Uz.size(), [&]() { return dis(gen); });
    std::generate(Ur.data(), Ur.data() + Ur.size(), [&]() { return dis(gen); });
    std::generate(Uh.data(), Uh.data() + Uh.size(), [&]() { return dis(gen); });

    bz.fill(0.0f);
    br.fill(0.0f);
    bh.fill(0.0f);
    h_prev.fill(0.0f);
}

Tensor GRU::forward(const Tensor& input) {
    if (input.shape().size() != 3 || input.shape()[1] != input_size) {
        throw std::invalid_argument("Input must be 3D with shape {batch_size, input_size, sequence_length}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    size_t seq_length = input.shape()[2];
    Tensor output({batch_size, hidden_size, seq_length});
    float* output_data = output.data();
    const float* input_data = input.data();
    float* h_prev_data = h_prev.data();

    for (size_t b = 0; b < batch_size; ++b) {
        Tensor h_current({hidden_size});
        std::copy(h_prev_data, h_prev_data + hidden_size, h_current.data());
        for (size_t t = 0; t < seq_length; ++t) {
            size_t input_idx = b * input_size * seq_length + t * input_size;
            Tensor xt({input_size});
            std::copy(input_data + input_idx, input_data + input_idx + input_size, xt.data());

            Tensor zt = sigmoid(xt.dot(Wz) + h_current.dot(Uz) + bz); // Update gate
            Tensor rt = sigmoid(xt.dot(Wr) + h_current.dot(Ur) + br); // Reset gate
            Tensor ht_candidate = tanh(xt.dot(Wh) + (rt * h_current).dot(Uh) + bh); // Candidate hidden
            Tensor ht = (Tensor({hidden_size}, 1.0f) - zt) * ht_candidate + zt * h_current; // New hidden state

            float* ht_data = ht.data();
            size_t output_idx = b * hidden_size * seq_length + t * hidden_size;
            std::copy(ht_data, ht_data + hidden_size, output_data + output_idx);
            std::copy(ht_data, ht_data + hidden_size, h_current.data());

            if (b == 0 && t == seq_length - 1) {
                std::copy(h_current.data(), h_current.data() + hidden_size, h_prev_data);
                zt_cache = zt;
                rt_cache = rt;
                ht_cache = ht_candidate;
            }
        }
    }

    return output;
}

Tensor GRU::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 3 || grad_output.shape()[1] != hidden_size) {
        throw std::invalid_argument("Gradient must be 3D with shape {batch_size, hidden_size, sequence_length}");
    }
    size_t batch_size = grad_output.shape()[0];
    size_t seq_length = grad_output.shape()[2];
    Tensor grad_input({batch_size, input_size, seq_length});
    grad_input.fill(0.0f);

    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    float* wz_data = Wz.data();
    float* wr_data = Wr.data();
    float* wh_data = Wh.data();
    float* uz_data = Uz.data();
    float* ur_data = Ur.data();
    float* uh_data = Uh.data();
    float* bz_data = bz.data();
    float* br_data = br.data();
    float* bh_data = bh.data();

    Tensor grad_Wz(Wz.shape()), grad_Wr(Wr.shape()), grad_Wh(Wh.shape());
    Tensor grad_Uz(Uz.shape()), grad_Ur(Ur.shape()), grad_Uh(Uh.shape());
    Tensor grad_bz(bz.shape()), grad_br(br.shape()), grad_bh(bh.shape());
    grad_Wz.fill(0.0f); grad_Wr.fill(0.0f); grad_Wh.fill(0.0f);
    grad_Uz.fill(0.0f); grad_Ur.fill(0.0f); grad_Uh.fill(0.0f);
    grad_bz.fill(0.0f); grad_br.fill(0.0f); grad_bh.fill(0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
        Tensor h_current({hidden_size});
        std::copy(h_prev.data(), h_prev.data() + hidden_size, h_current.data());
        Tensor grad_h_next({hidden_size});
        grad_h_next.fill(0.0f);

        for (int t = seq_length - 1; t >= 0; --t) {
            size_t input_idx = b * input_size * seq_length + t * input_size;
            size_t grad_idx = b * hidden_size * seq_length + t * hidden_size;
            Tensor xt({input_size});
            std::copy(input_cache.data() + input_idx, input_cache.data() + input_idx + input_size, xt.data());

            Tensor grad_ht({hidden_size});
            float* grad_ht_data = grad_ht.data();
            std::copy(grad_output_data + grad_idx, grad_output_data + grad_idx + hidden_size, grad_ht_data);
            for (size_t i = 0; i < hidden_size; ++i) {
                grad_ht_data[i] += grad_h_next({i});
            }

            Tensor zt = sigmoid(xt.dot(Wz) + h_current.dot(Uz) + bz);
            Tensor rt = sigmoid(xt.dot(Wr) + h_current.dot(Ur) + br);
            Tensor ht_candidate = tanh(xt.dot(Wh) + (rt * h_current).dot(Uh) + bh);
            Tensor ht = (Tensor({hidden_size}, 1.0f) - zt) * ht_candidate + zt * h_current;

            const float* zt_data = zt.data();
            const float* rt_data = rt.data();
            const float* ht_cand_data = ht_candidate.data();
            const float* h_curr_data = h_current.data();

            Tensor grad_zt({hidden_size});
            Tensor grad_ht_cand({hidden_size});
            Tensor grad_rt({hidden_size});
            float* grad_zt_data = grad_zt.data();
            float* grad_ht_cand_data = grad_ht_cand.data();
            float* grad_rt_data = grad_rt.data();

            for (size_t i = 0; i < hidden_size; ++i) {
                grad_zt_data[i] = grad_ht_data[i] * (h_curr_data[i] - ht_cand_data[i]) * zt_data[i] * (1.0f - zt_data[i]);
                grad_ht_cand_data[i] = grad_ht_data[i] * (1.0f - zt_data[i]) * (1.0f - ht_cand_data[i] * ht_cand_data[i]);
                grad_rt_data[i] = 0.0f; // Computed below
            }

            Tensor rt_h_curr = rt * h_current;
            Tensor grad_h_reset = grad_ht_cand.dot(Uh.transpose());
            for (size_t i = 0; i < hidden_size; ++i) {
                grad_rt_data[i] = grad_h_reset({i}) * h_curr_data[i] * rt_data[i] * (1.0f - rt_data[i]);
            }

            Tensor grad_xt = grad_zt.dot(Wz.transpose()) + grad_rt.dot(Wr.transpose()) + grad_ht_cand.dot(Wh.transpose());
            Tensor grad_h_prev = grad_zt.dot(Uz.transpose()) + grad_rt.dot(Ur.transpose()) + grad_ht_cand * rt.dot(Uh.transpose()) + grad_ht * zt;

            float* grad_xt_data = grad_xt.data();
            for (size_t i = 0; i < input_size; ++i) {
                grad_input_data[b * input_size * seq_length + t * input_size + i] = grad_xt_data[i];
            }

            for (size_t i = 0; i < hidden_size; ++i) {
                for (size_t j = 0; j < input_size; ++j) {
                    grad_Wz({j, i}) += grad_zt_data[i] * xt({j});
                    grad_Wr({j, i}) += grad_rt_data[i] * xt({j});
                    grad_Wh({j, i}) += grad_ht_cand_data[i] * xt({j});
                }
                for (size_t j = 0; j < hidden_size; ++j) {
                    grad_Uz({j, i}) += grad_zt_data[i] * h_curr_data[j];
                    grad_Ur({j, i}) += grad_rt_data[i] * h_curr_data[j];
                    grad_Uh({j, i}) += grad_ht_cand_data[i] * rt_h_curr({j});
                }
                grad_bz({i}) += grad_zt_data[i];
                grad_br({i}) += grad_rt_data[i];
                grad_bh({i}) += grad_ht_cand_data[i];
            }

            std::copy(grad_h_prev.data(), grad_h_prev.data() + hidden_size, grad_h_next.data());
            std::copy(ht.data(), ht.data() + hidden_size, h_current.data());
        }
    }

    float lr = learning_rate;
    for (size_t i = 0; i < Wz.size(); ++i) wz_data[i] -= lr * grad_Wz.data()[i];
    for (size_t i = 0; i < Wr.size(); ++i) wr_data[i] -= lr * grad_Wr.data()[i];
    for (size_t i = 0; i < Wh.size(); ++i) wh_data[i] -= lr * grad_Wh.data()[i];
    for (size_t i = 0; i < Uz.size(); ++i) uz_data[i] -= lr * grad_Uz.data()[i];
    for (size_t i = 0; i < Ur.size(); ++i) ur_data[i] -= lr * grad_Ur.data()[i];
    for (size_t i = 0; i < Uh.size(); ++i) uh_data[i] -= lr * grad_Uh.data()[i];
    for (size_t i = 0; i < bz.size(); ++i) bz_data[i] -= lr * grad_bz.data()[i];
    for (size_t i = 0; i < br.size(); ++i) br_data[i] -= lr * grad_br.data()[i];
    for (size_t i = 0; i < bh.size(); ++i) bh_data[i] -= lr * grad_bh.data()[i];

    return grad_input;
}

Tensor GRU::sigmoid(const Tensor& x) const {
    Tensor result(x.shape());
    float* result_data = result.data();
    const float* x_data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-x_data[i]));
    }
    return result;
}

Tensor GRU::tanh(const Tensor& x) const {
    Tensor result(x.shape());
    float* result_data = result.data();
    const float* x_data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        result_data[i] = std::tanh(x_data[i]);
    }
    return result;
}

void GRU::setWeights(const Tensor& Wz, const Tensor& Wr, const Tensor& Wh,
                     const Tensor& Uz, const Tensor& Ur, const Tensor& Uh,
                     const Tensor& bz, const Tensor& br, const Tensor& bh) {
    if (Wz.shape() != this->Wz.shape() || Wr.shape() != this->Wr.shape() || Wh.shape() != this->Wh.shape() ||
        Uz.shape() != this->Uz.shape() || Ur.shape() != this->Ur.shape() || Uh.shape() != this->Uh.shape() ||
        bz.shape() != this->bz.shape() || br.shape() != this->br.shape() || bh.shape() != this->bh.shape()) {
        throw std::invalid_argument("Shape mismatch in GRU weights or biases");
    }
    this->Wz = Wz;
    this->Wr = Wr;
    this->Wh = Wh;
    this->Uz = Uz;
    this->Ur = Ur;
    this->Uh = Uh;
    this->bz = bz;
    this->br = br;
    this->bh = bh;
}

void GRU::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    file.write(reinterpret_cast<const char*>(Wz.data()), Wz.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Wr.data()), Wr.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Wh.data()), Wh.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Uz.data()), Uz.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Ur.data()), Ur.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(Uh.data()), Uh.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bz.data()), bz.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(br.data()), br.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_prev.data()), h_prev.size() * sizeof(float));
}

std::unique_ptr<GRU> GRU::load(std::ifstream& file) {
    size_t input_size, hidden_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    auto layer = std::make_unique<GRU>(input_size, hidden_size);
    file.read(reinterpret_cast<char*>(layer->Wz.data()), layer->Wz.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Wr.data()), layer->Wr.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Wh.data()), layer->Wh.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Uz.data()), layer->Uz.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Ur.data()), layer->Ur.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->Uh.data()), layer->Uh.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bz.data()), layer->bz.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->br.data()), layer->br.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->bh.data()), layer->bh.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->h_prev.data()), layer->h_prev.size() * sizeof(float));
    return layer;
}

void GRU::print() const {
    std::cout << "GRU Layer: input_size=" << input_size << ", hidden_size=" << hidden_size << "\n";
    std::cout << "Wz:\n"; Wz.print();
    std::cout << "Wr:\n"; Wr.print();
    std::cout << "Wh:\n"; Wh.print();
    std::cout << "Uz:\n"; Uz.print();
    std::cout << "Ur:\n"; Ur.print();
    std::cout << "Uh:\n"; Uh.print();
    std::cout << "bz:\n"; bz.print();
    std::cout << "br:\n"; br.print();
    std::cout << "bh:\n"; bh.print();
    std::cout << "h_prev:\n"; h_prev.print();
}
