#ifndef IDENTITY_LAYER_H
#define IDENTITY_LAYER_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class IdentityLayer : public Layer {
public:
    explicit IdentityLayer(const std::vector<size_t>& input_shape);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const std::vector<size_t>& getInputShape() const { return input_shape; }

    void save(std::ofstream& file) const ;
    static std::unique_ptr<IdentityLayer> load(std::ifstream& file);
    void print() const;

private:
    std::vector<size_t> input_shape; // Expected input shape
};

#endif // IDENTITY_LAYER_H
