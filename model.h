#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <memory>
#include "tensor.h"
#include "layers/layer.h"

class Model {
public:
    // Добавить слой в модель
    void addLayer(std::shared_ptr<Layer> layer);

    // Прямой проход: вычисляет выходные данные на основе входных
    Tensor predict(const Tensor& input);

    // Обучение модели
    void train(const Tensor& input, const Tensor& target, size_t epochs, float learning_rate);

    // Получить слои модели
    const std::vector<std::shared_ptr<Layer>>& getLayers() const;

private:
    std::vector<std::shared_ptr<Layer>> layers; // Слои модели
};

#endif // MODEL_H
