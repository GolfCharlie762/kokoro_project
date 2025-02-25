#ifndef DROPOUT_H
#define DROPOUT_H

#include "math/tensor.h"
#include "layer.h"
#include <random>

class Dropout : public Layer {
public:
    Dropout(float rate);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    float getRate() const;
    void setRate(float rate);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Dropout> load(std::ifstream& file);
    void print() const;

private:
    float rate;
    Tensor mask;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

#endif // DROPOUT_H
