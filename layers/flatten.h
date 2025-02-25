#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"
#include "math/tensor.h"

class Flatten : public Layer {
public:
    Flatten();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Flatten> load(std::ifstream& file);
    void print() const;

private:
    std::vector<size_t> input_shape; // Store original input shape for backward pass
};

#endif // FLATTEN_H
