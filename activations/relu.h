#ifndef RELU_H
#define RELU_H

#include "math/tensor.h"
#include "layers/layer.h"

class ReLU : public Layer {
public:
    ReLU();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<ReLU> load(std::ifstream& file);
    void print() const;

private:
    Tensor input_cache;
};

#endif // RELU_H
