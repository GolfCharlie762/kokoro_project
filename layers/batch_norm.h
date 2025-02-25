#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "math/tensor.h"
#include "layer.h"

class BatchNorm : public Layer {
public:
    BatchNorm(size_t num_features, float epsilon = 1e-5f, float momentum = 0.9f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<BatchNorm> load(std::ifstream& file);
    void print() const;

private:
    size_t num_features;
    float epsilon;
    float momentum;

    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    Tensor input_cache;
    Tensor normalized;
};

#endif // BATCH_NORM_H
