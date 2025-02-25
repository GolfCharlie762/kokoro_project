#include "dropout.h"
#include <fstream>

Dropout::Dropout(float rate)
    : rate(rate), mask(Tensor()), gen(std::random_device{}()), dist(0.0f, 1.0f) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    if (input.shape().size() < 1) {
        throw std::invalid_argument("Input must have at least one dimension");
    }
    mask = Tensor(input.shape());
    float* mask_data = mask.data();
    const float* input_data = input.data();
    Tensor output(input.shape());
    float* output_data = output.data();
    size_t size = input.size();
    float scale = 1.0f / (1.0f - rate);

    for (size_t i = 0; i < size; ++i) {
        mask_data[i] = (dist(gen) < rate) ? 0.0f : scale;
        output_data[i] = input_data[i] * mask_data[i];
    }

    return output;
}

Tensor Dropout::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != mask.shape()) {
        throw std::invalid_argument("Gradient shape must match mask shape");
    }
    Tensor grad_input(grad_output.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* mask_data = mask.data();
    size_t size = grad_output.size();

    for (size_t i = 0; i < size; ++i) {
        grad_input_data[i] = grad_output_data[i] * mask_data[i];
    }

    return grad_input;
}

float Dropout::getRate() const {
    return rate;
}

void Dropout::setRate(float rate) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
    this->rate = rate;
}

void Dropout::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&rate), sizeof(rate));
    size_t shape_size = mask.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(mask.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(mask.data()), mask.size() * sizeof(float));
}

std::unique_ptr<Dropout> Dropout::load(std::ifstream& file) {
    float rate;
    file.read(reinterpret_cast<char*>(&rate), sizeof(rate));
    auto layer = std::make_unique<Dropout>(rate);
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->mask = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->mask.data()), layer->mask.size() * sizeof(float));
    return layer;
}

void Dropout::print() const {
    std::cout << "Dropout Layer: rate=" << rate << "\n";
    std::cout << "Mask:\n";
    mask.print();
}
