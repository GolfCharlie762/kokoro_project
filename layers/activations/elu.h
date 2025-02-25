#ifndef ELU_H
#define ELU_H

#include "math/tensor.h"
#include "layers/layer.h"

class ELU : public Layer {
public:
    explicit ELU(float alpha = 1.0f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void save(std::ofstream& file) const ;
    static std::unique_ptr<ELU> load(std::ifstream& file);
    void print() const;

private:
    Tensor input_cache;
    float alpha;
};

#endif // ELU_H
