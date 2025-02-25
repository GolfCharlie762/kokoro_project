#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layers/layer.h"
#include "math/tensor.h"

class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& getWeights() const { return weights; }
    const Tensor& getBiases() const { return biases; }
    size_t getInputSize() const { return input_size; }
    size_t getOutputSize() const { return output_size; }

    void setWeights(const Tensor& w);
    void setBiases(const Tensor& b);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<DenseLayer> load(std::ifstream& file);

    void print() const;

private:
    size_t input_size;
    size_t output_size;
    Tensor weights;
    Tensor biases;
    Tensor input_cache;
};

#endif // DENSE_LAYER_H
