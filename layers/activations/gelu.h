#ifndef GELU_H
#define GELU_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class GELU : public Layer {
public:
    GELU();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<GELU> load(std::ifstream& file);
    void print() const;

private:
    Tensor input_cache; // Cached input for backward pass
    float gelu(float x) const;
    float gelu_derivative(float x) const;
};

#endif // GELU_H
