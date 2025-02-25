#include "conv2d.h"
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>

Conv2D::Conv2D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride, size_t padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size), stride(stride), padding(padding),
      kernels({output_channels, input_channels, kernel_size, kernel_size}), biases({output_channels}), input_cache(Tensor()) {
    float limit = std::sqrt(6.0f / (input_channels * kernel_size * kernel_size + output_channels * kernel_size * kernel_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);
    std::generate(kernels.data(), kernels.data() + kernels.size(), [&]() { return dis(gen); });
    biases.fill(0.0f);
}

Tensor Conv2D::forward(const Tensor& input) {
    if (input.shape().size() != 4 || input.shape()[1] != input_channels) {
        throw std::invalid_argument("Input must be 4D with shape {batch_size, input_channels, height, width}");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    size_t input_height = input.shape()[2];
    size_t input_width = input.shape()[3];
    size_t padded_height = input_height + 2 * padding;
    size_t padded_width = input_width + 2 * padding;
    size_t output_height = (padded_height - kernel_size) / stride + 1;
    size_t output_width = (padded_width - kernel_size) / stride + 1;

    Tensor output({batch_size, output_channels, output_height, output_width});
    Tensor padded_input = pad(input);
    float* output_data = output.data();
    const float* padded_data = padded_input.data();
    const float* kernels_data = kernels.data();
    const float* biases_data = biases.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < output_channels; ++oc) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float sum = biases_data[oc];
                    for (size_t ic = 0; ic < input_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                size_t ih = oh * stride + kh;
                                size_t iw = ow * stride + kw;
                                size_t input_idx = b * input_channels * padded_height * padded_width +
                                                  ic * padded_height * padded_width + ih * padded_width + iw;
                                size_t kernel_idx = oc * input_channels * kernel_size * kernel_size +
                                                   ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                sum += padded_data[input_idx] * kernels_data[kernel_idx];
                            }
                        }
                    }
                    output_data[b * output_channels * output_height * output_width +
                               oc * output_height * output_width + oh * output_width + ow] = sum;
                }
            }
        }
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 4 || grad_output.shape()[1] != output_channels) {
        throw std::invalid_argument("Gradient must be 4D with shape {batch_size, output_channels, height, width}");
    }
    size_t batch_size = input_cache.shape()[0];
    size_t input_height = input_cache.shape()[2];
    size_t input_width = input_cache.shape()[3];
    size_t output_height = grad_output.shape()[2];
    size_t output_width = grad_output.shape()[3];

    Tensor grad_kernels(kernels.shape());
    Tensor grad_biases(biases.shape());
    Tensor grad_input({batch_size, input_channels, input_height, input_width});
    grad_input.fill(0.0f);
    Tensor padded_input = pad(input_cache);

    float* grad_kernels_data = grad_kernels.data();
    float* grad_biases_data = grad_biases.data();
    float* grad_input_data = grad_input.data();
    float* kernels_data = kernels.data();
    float* biases_data = biases.data();
    const float* grad_output_data = grad_output.data();
    const float* padded_data = padded_input.data();

    // Compute gradients for biases
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < output_channels; ++oc) {
            float sum = 0.0f;
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    sum += grad_output_data[b * output_channels * output_height * output_width +
                                           oc * output_height * output_width + oh * output_width + ow];
                }
            }
            grad_biases_data[oc] += sum;
        }
    }

    // Compute gradients for kernels
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < output_channels; ++oc) {
            for (size_t ic = 0; ic < input_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        float sum = 0.0f;
                        for (size_t oh = 0; oh < output_height; ++oh) {
                            for (size_t ow = 0; ow < output_width; ++ow) {
                                size_t ih = oh * stride + kh;
                                size_t iw = ow * stride + kw;
                                size_t input_idx = b * input_channels * (input_height + 2 * padding) * (input_width + 2 * padding) +
                                                  ic * (input_height + 2 * padding) * (input_width + 2 * padding) + ih * (input_width + 2 * padding) + iw;
                                size_t grad_idx = b * output_channels * output_height * output_width +
                                                 oc * output_height * output_width + oh * output_width + ow;
                                sum += padded_data[input_idx] * grad_output_data[grad_idx];
                            }
                        }
                        size_t kernel_idx = oc * input_channels * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        grad_kernels_data[kernel_idx] += sum;
                    }
                }
            }
        }
    }

    // Compute gradient for input
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < output_channels; ++oc) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                for (size_t ow = 0; ow < output_width; ++ow) {
                    for (size_t ic = 0; ic < input_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                size_t ih = oh * stride + kh - padding;
                                size_t iw = ow * stride + kw - padding;
                                if (ih < input_height && iw < input_width) {
                                    size_t grad_idx = b * output_channels * output_height * output_width +
                                                     oc * output_height * output_width + oh * output_width + ow;
                                    size_t input_idx = b * input_channels * input_height * input_width +
                                                      ic * input_height * input_width + ih * input_width + iw;
                                    size_t kernel_idx = oc * input_channels * kernel_size * kernel_size +
                                                       ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    grad_input_data[input_idx] += kernels_data[kernel_idx] * grad_output_data[grad_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Update weights
    float lr = learning_rate / batch_size;
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernels_data[i] -= lr * grad_kernels_data[i];
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        biases_data[i] -= lr * grad_biases_data[i];
    }

    return grad_input;
}

Tensor Conv2D::pad(const Tensor& input) const {
    if (padding == 0) return input;
    size_t batch_size = input.shape()[0];
    Tensor padded({batch_size, input_channels, input.shape()[2] + 2 * padding, input.shape()[3] + 2 * padding});
    padded.fill(0.0f);
    float* padded_data = padded.data();
    const float* input_data = input.data();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < input_channels; ++c) {
            for (size_t h = 0; h < input.shape()[2]; ++h) {
                for (size_t w = 0; w < input.shape()[3]; ++w) {
                    size_t input_idx = b * input_channels * input.shape()[2] * input.shape()[3] +
                                      c * input.shape()[2] * input.shape()[3] + h * input.shape()[3] + w;
                    size_t padded_idx = b * input_channels * (input.shape()[2] + 2 * padding) * (input.shape()[3] + 2 * padding) +
                                       c * (input.shape()[2] + 2 * padding) * (input.shape()[3] + 2 * padding) +
                                       (h + padding) * (input.shape()[3] + 2 * padding) + (w + padding);
                    padded_data[padded_idx] = input_data[input_idx];
                }
            }
        }
    }
    return padded;
}

void Conv2D::setWeights(const Tensor& new_kernels, const Tensor& new_biases) {
    if (new_kernels.shape() != kernels.shape() || new_biases.shape() != biases.shape()) {
        throw std::invalid_argument("Shape mismatch in kernels or biases");
    }
    kernels = new_kernels;
    biases = new_biases;
}

void Conv2D::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&input_channels), sizeof(input_channels));
    file.write(reinterpret_cast<const char*>(&output_channels), sizeof(output_channels));
    file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
    file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    file.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
    file.write(reinterpret_cast<const char*>(kernels.data()), kernels.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

std::unique_ptr<Conv2D> Conv2D::load(std::ifstream& file) {
    size_t input_channels, output_channels, kernel_size, stride, padding;
    file.read(reinterpret_cast<char*>(&input_channels), sizeof(input_channels));
    file.read(reinterpret_cast<char*>(&output_channels), sizeof(output_channels));
    file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    file.read(reinterpret_cast<char*>(&padding), sizeof(padding));
    auto layer = std::make_unique<Conv2D>(input_channels, output_channels, kernel_size, stride, padding);
    file.read(reinterpret_cast<char*>(layer->kernels.data()), layer->kernels.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(layer->biases.data()), layer->biases.size() * sizeof(float));
    return layer;
}

void Conv2D::print() const {
    std::cout << "Conv2D Layer: input_channels=" << input_channels << ", output_channels=" << output_channels
              << ", kernel_size=" << kernel_size << ", stride=" << stride << ", padding=" << padding << "\n";
    std::cout << "Kernels:\n"; kernels.print();
    std::cout << "Biases:\n"; biases.print();
}
