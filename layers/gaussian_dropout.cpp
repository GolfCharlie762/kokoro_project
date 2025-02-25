#include "gaussian_dropout.h"
#include <fstream>

GaussianDropout::GaussianDropout(float rate, bool training)
    : rate(rate), training(training), noise(Tensor()), gen(std::random_device{}()),
      dist(1.0f, std::sqrt(rate / (1.0f - rate))) { // Mean 1, variance rate/(1-rate)
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Gaussian dropout rate must be in [0, 1)");
    }
}

Tensor GaussianDropout::forward(const Tensor& input) {
    if (input.shape().size() < 1) {
        throw std::invalid_argument("Input must have at least one dimension");
    }
    if (!training) return input; // No dropout during inference

    noise = Tensor(input.shape());
    float* noise_data = noise.data();
    const float* input_data = input.data();
    Tensor output(input.shape());
    float* output_data = output.data();
    size_t size = input.size();

    for (size_t i = 0; i < size; ++i) {
        noise_data[i] = dist(gen);
        output_data[i] = input_data[i] * noise_data[i];
    }

    return output;
}

Tensor GaussianDropout::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape() != noise.shape()) {
        throw std::invalid_argument("Gradient shape must match noise shape");
    }
    if (!training) return grad_output; // No modification during inference

    Tensor grad_input(grad_output.shape());
    float* grad_input_data = grad_input.data();
    const float* grad_output_data = grad_output.data();
    const float* noise_data = noise.data();
    size_t size = grad_output.size();

    for (size_t i = 0; i < size; ++i) {
        grad_input_data[i] = grad_output_data[i] * noise_data[i];
    }

    return grad_input;
}

float GaussianDropout::getRate() const {
    return rate;
}

void GaussianDropout::setRate(float rate) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Gaussian dropout rate must be in [0, 1)");
    }
    this->rate = rate;
    dist = std::normal_distribution<float>(1.0f, std::sqrt(rate / (1.0f - rate)));
}

bool GaussianDropout::isTraining() const {
    return training;
}

void GaussianDropout::setTraining(bool training) {
    this->training = training;
}

void GaussianDropout::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&rate), sizeof(rate));
    file.write(reinterpret_cast<const char*>(&training), sizeof(training));
    size_t shape_size = noise.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(noise.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(noise.data()), noise.size() * sizeof(float));
}

std::unique_ptr<GaussianDropout> GaussianDropout::load(std::ifstream& file) {
    float rate;
    bool training;
    file.read(reinterpret_cast<char*>(&rate), sizeof(rate));
    file.read(reinterpret_cast<char*>(&training), sizeof(training));
    auto layer = std::make_unique<GaussianDropout>(rate, training);
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->noise = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->noise.data()), layer->noise.size() * sizeof(float));
    return layer;
}

void GaussianDropout::print() const {
    std::cout << "GaussianDropout Layer: rate=" << rate << ", training=" << (training ? "true" : "false") << "\n";
    std::cout << "Noise:\n";
    noise.print();
}
