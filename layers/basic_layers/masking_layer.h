#ifndef MASKING_LAYER_H
#define MASKING_LAYER_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class MaskingLayer : public Layer {
public:
    MaskingLayer(const std::vector<size_t>& input_shape, float mask_value = 0.0f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    float getMaskValue() const { return mask_value; }
    const std::vector<size_t>& getInputShape() const { return input_shape; }
    const Tensor& getMask() const { return mask; }

    void save(std::ofstream& file) const ;
    static std::unique_ptr<MaskingLayer> load(std::ifstream& file);
    void print() const;

private:
    std::vector<size_t> input_shape; // Expected input shape {batch_size, sequence_length, features}
    float mask_value;                // Value to mask (e.g., 0.0 for padding)
    Tensor mask;                     // Binary mask: 1.0 for unmasked, 0.0 for masked
    Tensor input_cache;              // Cached input for backward pass
};

#endif // MASKING_LAYER_H
