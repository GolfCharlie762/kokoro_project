#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;

    // Прямой проход
    virtual Tensor forward(const Tensor& input) = 0;

    // Обратный проход
    virtual Tensor backward(const Tensor& grad_output, float learning_rate) = 0;

    // Сохранение слоя в файл
    virtual void save(std::ofstream& file) const = 0;

    // Загрузка слоя из файла
    static std::shared_ptr<Layer> load(std::ifstream& file);

    // Получить имя слоя
    virtual std::string getName() const = 0;

    // Получить форму входных данных
    virtual std::vector<size_t> getInputShape() const = 0;

    // Получить форму выходных данных
    virtual std::vector<size_t> getOutputShape() const = 0;

    // Получить количество параметров
    virtual size_t getNumParameters() const = 0;
};

#endif // LAYER_H
