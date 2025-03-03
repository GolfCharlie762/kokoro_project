#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class NormalizationLayer : public Layer {
public:
    NormalizationLayer(const std::vector<size_t>& input_shape, float epsilon = 1e-8f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void adapt(const Tensor& data);
    const Tensor& getMean() const { return mean; }
    const Tensor& getVariance() const { return variance; }
    float getEpsilon() const { return epsilon; }

    void save(std::ofstream& file) const ;
    static std::unique_ptr<NormalizationLayer> load(std::ifstream& file);
    void print() const;

private:
    std::vector<size_t> input_shape; // Expected input shape
    Tensor mean;                     // Precomputed mean per feature
    Tensor variance;                 // Precomputed variance per feature
    float epsilon;                   // Small constant to avoid division by zero
    Tensor input_cache;              // Cached input for backward pass

    void computeStatistics(const Tensor& data);
};

#endif // NORMALIZATION_H
