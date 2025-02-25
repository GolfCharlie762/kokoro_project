#ifndef SIGMOID_H
#define SIGMOID_H

#include "math/tensor.h"
#include "layers/layer.h"

class Sigmoid : public Layer {
public:
    Sigmoid();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Sigmoid> load(std::ifstream& file);
    void print() const;

private:
    Tensor output_cache;
};

#endif // SIGMOID_H
