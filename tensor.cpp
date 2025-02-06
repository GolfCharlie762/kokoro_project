#include "tensor.h"
#include <random>
#include <numeric>
#include <functional>
// Конструктор
Tensor::Tensor(const std::vector<size_t>& shape) : _shape(shape) {
    // Вычисляем общее количество элементов в тензоре
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    _data.resize(total_size, 0.0f); // Инициализируем данные нулями
}

// Доступ к элементам тензора по индексам (неконстантная версия)
float& Tensor::operator()(const std::vector<size_t>& indices) {
    assert(indices.size() == _shape.size()); // Проверяем, что количество индексов совпадает с размерностью тензора
    size_t index = 0;
    size_t stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= _shape[i];
    }
    return _data[index];
}

// Доступ к элементам тензора по индексам (константная версия)
const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    assert(indices.size() == _shape.size());
    size_t index = 0;
    size_t stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= _shape[i];
    }
    return _data[index];
}

// Получить форму тензора
std::vector<size_t> Tensor::shape() const {
    return _shape;
}

// Получить общее количество элементов в тензоре
size_t Tensor::size() const {
    return _data.size();
}

// Заполнить тензор значением
void Tensor::fill(float value) {
    std::fill(_data.begin(), _data.end(), value);
}

// Заполнить тензор случайными значениями
void Tensor::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (float& val : _data) {
        val = dist(gen);
    }
}

// Вывести тензор (для отладки)
void Tensor::print() const {
    std::cout << "Tensor shape: (";
    for (size_t dim : _shape) {
        std::cout << dim << ", ";
    }
    std::cout << ")\n";

    // Рекурсивно выводим элементы тензора
    std::function<void(const Tensor&, std::vector<size_t>, size_t)> print_recursive;
    print_recursive = [&](const Tensor& tensor, std::vector<size_t> indices, size_t dim) {
        if (dim == tensor.shape().size()) {
            std::cout << tensor(indices) << " ";
            return;
        }
        for (size_t i = 0; i < tensor.shape()[dim]; ++i) {
            indices[dim] = i;
            print_recursive(tensor, indices, dim + 1);
        }
        if (dim == tensor.shape().size() - 1) {
            std::cout << "\n";
        }
    };

    std::vector<size_t> indices(_shape.size(), 0);
    print_recursive(*this, indices, 0);
}

// Перегрузка оператора вычитания (Tensor - Tensor)
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");
    }

    Tensor result(shape());
    for (size_t i = 0; i < size(); ++i) {
        result({i}) = _data[i] - other({i});
    }
    return result;
}

// Перегрузка оператора вычитания (int - Tensor)
Tensor operator-(int value, const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result({i}) = value - tensor({i});
    }
    return result;
}


// Скалярное произведение
Tensor Tensor::dot(const Tensor& other) const {
    if (shape().size() != 2 || other.shape().size() != 2 || shape()[1] != other.shape()[0]) {
        throw std::invalid_argument("Tensors must be 2D and have compatible shapes for dot product.");
    }

    Tensor result({shape()[0], other.shape()[1]});
    for (size_t i = 0; i < shape()[0]; ++i) {
        for (size_t j = 0; j < other.shape()[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < shape()[1]; ++k) {
                sum += (*this)({i, k}) * other({k, j});
            }
            result({i, j}) = sum;
        }
    }
    return result;
}

// Транспонирование
Tensor Tensor::transpose() const {
    if (shape().size() != 2) {
        throw std::invalid_argument("Tensor must be 2D for transpose.");
    }

    Tensor result({shape()[1], shape()[0]});
    for (size_t i = 0; i < shape()[0]; ++i) {
        for (size_t j = 0; j < shape()[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

// Перегрузка оператора умножения
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication.");
    }

    Tensor result(shape());
    for (size_t i = 0; i < size(); ++i) {
        result({i}) = _data[i] * other({i});
    }
    return result;
}

// Перегрузка оператора сложения
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape() != other.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }

    Tensor result(shape());
    for (size_t i = 0; i < size(); ++i) {
        result({i}) = _data[i] + other({i});
    }
    return result;
}







