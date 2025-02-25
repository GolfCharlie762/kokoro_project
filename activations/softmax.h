#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "math/tensor.h"
#include "layers/layer.h"

class Softmax : public Layer {
public:
    Softmax();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Softmax> load(std::ifstream& file);
    void print() const;

private:
    Tensor output_cache;
};

#endif // SOFTMAX_H
