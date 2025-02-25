#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include "math/tensor.h" // Assuming tensor.h is in math/ folder based on your prior structure
#include "layers/layer.h"

class MaxPooling2D : public Layer {
public:
    MaxPooling2D(size_t pool_size, size_t stride = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<MaxPooling2D> load(std::ifstream& file);
    void print() const;

private:
    size_t pool_size, stride;
    Tensor input_cache;
    Tensor max_indices; // Store indices of max values for backpropagation
};

#endif // MAX_POOLING2D_H
