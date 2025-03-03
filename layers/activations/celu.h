#ifndef CELU_H
#define CELU_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class CELU : public Layer {
public:
    explicit CELU(float alpha = 1.0f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    float getAlpha() const { return alpha; }
    void setAlpha(float new_alpha);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<CELU> load(std::ifstream& file);
    void print() const;

private:
    float alpha;        // Alpha parameter for CELU
    Tensor input_cache; // Cached input for backward pass
    float celu(float x) const;
    float celu_derivative(float x) const;
};

#endif // CELU_H
