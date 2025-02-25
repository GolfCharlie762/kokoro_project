#ifndef CONV1D_H
#define CONV1D_H

#include "math/tensor.h"
#include "layers/layer.h"

class Conv1D : public Layer {
public:
    Conv1D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);
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
    static std::unique_ptr<Conv1D> load(std::ifstream& file);
    void print() const;

private:
    size_t input_channels, output_channels, kernel_size, stride, padding;
    Tensor kernels; // Shape: {output_channels, input_channels, kernel_size}
    Tensor biases;  // Shape: {output_channels}
    Tensor input_cache;

    Tensor pad(const Tensor& input) const;
};

#endif // CONV1D_H
