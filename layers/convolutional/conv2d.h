#ifndef CONV2D_H
#define CONV2D_H

#include "math/tensor.h"
#include "layers/layer.h"

class Conv2D : public Layer {
public:
    Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& getKernels() const { return kernels; }
    const Tensor& getBiases() const { return biases; }
    size_t getInputChannels() const { return input_channels; }
    size_t getOutputChannels() const { return output_channels; }
    size_t getKernelSize() const { return kernel_size; }
    size_t getStride() const { return stride; }
    size_t getPadding() const { return padding; }

    void setWeights(const Tensor& new_kernels, const Tensor& new_biases);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Conv2D> load(std::ifstream& file);
    void print() const;

private:
    size_t input_channels, output_channels, kernel_size, stride, padding;
    Tensor kernels;
    Tensor biases;
    Tensor input_cache;

    Tensor pad(const Tensor& input) const;
};

#endif // CONV2D_H
