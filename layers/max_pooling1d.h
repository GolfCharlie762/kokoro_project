#ifndef MAX_POOLING1D_H
#define MAX_POOLING1D_H

#include "math/tensor.h"
#include "layer.h"

class MaxPooling1D : public Layer {
public:
    MaxPooling1D(size_t pool_size, size_t stride = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<MaxPooling1D> load(std::ifstream& file);
    void print() const;

private:
    size_t pool_size, stride;
    Tensor input_cache;
    Tensor max_indices; // Store indices of max values for backpropagation
};

#endif // MAX_POOLING1D_H
