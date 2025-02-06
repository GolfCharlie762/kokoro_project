#include "model.h"

// Добавить слой в модель
void Model::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

// Прямой проход
Tensor Model::predict(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

// Обучение модели
void Model::train(const Tensor& input, const Tensor& target, size_t epochs, float learning_rate) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Прямой проход
        Tensor output = predict(input);

        // Вычисление ошибки
        Tensor error = output - target;
        float loss = 0.0f;
        for (size_t i = 0; i < error.size(); ++i) {
            loss += error({i}) * error({i});
        }
        loss /= error.size();

        // Обратный проход
        Tensor grad_output = error;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad_output = (*it)->backward(grad_output, learning_rate);
        }

        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
}

// Получить слои модели
const std::vector<std::shared_ptr<Layer>>& Model::getLayers() const {
    return layers;
}
