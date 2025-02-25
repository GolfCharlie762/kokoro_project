#ifndef TANH_H
#define TANH_H

#include "math/tensor.h"
#include "layers/layer.h"

class Tanh : public Layer {
public:
    Tanh();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Tanh> load(std::ifstream& file);
    void print() const;

private:
    Tensor input_cache;
};

#endif // TANH_H
