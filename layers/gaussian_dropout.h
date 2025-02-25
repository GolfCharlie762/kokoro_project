#ifndef GAUSSIAN_DROPOUT_H
#define GAUSSIAN_DROPOUT_H

#include "math/tensor.h"
#include "layer.h"
#include <random>

class GaussianDropout : public Layer {
public:
    GaussianDropout(float rate, bool training = true);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    float getRate() const;
    void setRate(float rate);
    bool isTraining() const;
    void setTraining(bool training);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<GaussianDropout> load(std::ifstream& file);
    void print() const;

private:
    float rate;
    bool training;
    Tensor noise; // Gaussian noise mask
    std::mt19937 gen;
    std::normal_distribution<float> dist;
};

#endif // GAUSSIAN_DROPOUT_H
