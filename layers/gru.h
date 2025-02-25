#ifndef GRU_H
#define GRU_H

#include "layer.h"
#include "math/tensor.h"
#include <vector>

class GRU : public Layer {
public:
    GRU(size_t input_size, size_t hidden_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& getWz() const { return Wz; }
    const Tensor& getWr() const { return Wr; }
    const Tensor& getWh() const { return Wh; }
    const Tensor& getUz() const { return Uz; }
    const Tensor& getUr() const { return Ur; }
    const Tensor& getUh() const { return Uh; }
    const Tensor& getBz() const { return bz; }
    const Tensor& getBr() const { return br; }
    const Tensor& getBh() const { return bh; }
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }

    void setWeights(const Tensor& Wz, const Tensor& Wr, const Tensor& Wh,
                    const Tensor& Uz, const Tensor& Ur, const Tensor& Uh,
                    const Tensor& bz, const Tensor& br, const Tensor& bh);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<GRU> load(std::ifstream& file);
    void print() const;

private:
    size_t input_size;
    size_t hidden_size;

    Tensor Wz, Wr, Wh; // Input weights: update, reset, hidden
    Tensor Uz, Ur, Uh; // Recurrent weights: update, reset, hidden
    Tensor bz, br, bh; // Biases
    Tensor h_prev;     // Previous hidden state
    Tensor input_cache;// Cached input
    Tensor zt_cache;   // Cached update gate
    Tensor rt_cache;   // Cached reset gate
    Tensor ht_cache;   // Cached candidate hidden state

    Tensor sigmoid(const Tensor& x) const;
    Tensor tanh(const Tensor& x) const;
};

#endif // GRU_H
