#ifndef SWISH_H
#define SWISH_H

#include "math/tensor.h"
#include "layers/layer.h"

class Swish : public Layer {
public:
    Swish();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Swish> load(std::ifstream& file);
    void print() const;

private:
    Tensor input_cache;
};

#endif // SWISH_H
