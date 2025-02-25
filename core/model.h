#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>
#include "math/tensor.h"
#include "layers/layer.h"

class Optimizer;

class Model {
public:
    enum class LossType { MSE, CrossEntropy };
    Model(LossType loss_type = LossType::MSE);
    void addLayer(std::shared_ptr<Layer> layer);
    Tensor predict(const Tensor& input);
    void train(const Tensor& input, const Tensor& target, size_t epochs, float learning_rate, size_t batch_size = 1);
    float validate(const Tensor& input, const Tensor& target);
    const std::vector<std::shared_ptr<Layer>>& getLayers() const;
    void setOptimizer(std::unique_ptr<Optimizer> optimizer);

private:
    std::vector<std::shared_ptr<Layer>> layers;
    LossType loss_type_;
    std::unique_ptr<Optimizer> optimizer_;
    float compute_loss(const Tensor& output, const Tensor& target) const;
};

#endif // MODEL_H
