#ifndef LAMBDA_LAYER_H
#define LAMBDA_LAYER_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <functional>
#include <vector>

class LambdaLayer : public Layer {
public:
    LambdaLayer(std::function<float(float)> forward_fn,
                std::function<float(float)> backward_fn,
                const std::vector<size_t>& input_shape);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const std::vector<size_t>& getInputShape() const { return input_shape; }

    void save(std::ofstream& file) const ;
    static std::unique_ptr<LambdaLayer> load(std::ifstream& file);
    void print() const;

private:
    std::function<float(float)> forward_fn;   // Custom forward function
    std::function<float(float)> backward_fn;  // Custom backward function (derivative)
    std::vector<size_t> input_shape;          // Expected input shape
    Tensor input_cache;                       // Cached input for backward pass
};

#endif // LAMBDA_LAYER_H
