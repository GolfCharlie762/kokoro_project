#ifndef LSTM_H
#define LSTM_H

#include "math/tensor.h"
#include "layer.h"

class LSTM : public Layer {
public:
    LSTM(size_t input_size, size_t hidden_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& getWf() const { return Wf; }
    const Tensor& getWi() const { return Wi; }
    const Tensor& getWo() const { return Wo; }
    const Tensor& getWc() const { return Wc; }
    const Tensor& getBf() const { return bf; }
    const Tensor& getBi() const { return bi; }
    const Tensor& getBo() const { return bo; }
    const Tensor& getBc() const { return bc; }
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }

    void setWeights(const Tensor& wf, const Tensor& wi, const Tensor& wo, const Tensor& wc,
                    const Tensor& bf, const Tensor& bi, const Tensor& bo, const Tensor& bc);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<LSTM> load(std::ifstream& file);
    void print() const;

private:
    size_t input_size;
    size_t hidden_size;

    Tensor Wf, Wi, Wo, Wc;
    Tensor bf, bi, bo, bc;
    Tensor h_prev, c_prev;
    Tensor input_cache;

    Tensor sigmoid(const Tensor& x) const;
    Tensor tanh(const Tensor& x) const;
};

#endif // LSTM_H
