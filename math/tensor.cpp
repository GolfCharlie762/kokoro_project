#include "tensor.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>

Tensor::Tensor(const std::vector<size_t>& shape) : _shape(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    if (total_size == 0) throw std::invalid_argument("Tensor size cannot be zero");
    _data.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float value) : _shape(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    if (total_size == 0) throw std::invalid_argument("Tensor size cannot be zero");
    _data.resize(total_size, value);
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data) : _shape(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    if (total_size != data.size()) throw std::invalid_argument("Data size does not match shape");
    _data = data;
}

float& Tensor::operator()(const std::vector<size_t>& indices) {
    return _data[compute_index(indices)];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    return _data[compute_index(indices)];
}

size_t Tensor::compute_index(const std::vector<size_t>& indices) const {
    if (indices.size() != _shape.size()) throw std::invalid_argument("Index dimension mismatch");
    size_t index = 0, stride = 1;
    for (int i = _shape.size() - 1; i >= 0; --i) {
        if (indices[i] >= _shape[i]) throw std::out_of_range("Index out of bounds");
        index += indices[i] * stride;
        stride *= _shape[i];
    }
    return index;
}

void Tensor::fill(float value) {
    std::fill(_data.begin(), _data.end(), value);
}

void Tensor::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (float& val : _data) val = dist(gen);
}

Tensor Tensor::slice(size_t start, size_t end) const {
    if (_shape.empty()) throw std::runtime_error("Cannot slice empty tensor");
    if (start >= _shape[0] || end > _shape[0] || start >= end) {
        throw std::out_of_range("Invalid slice range");
    }
    size_t batch_size = end - start;
    std::vector<size_t> new_shape = _shape;
    new_shape[0] = batch_size;
    size_t elements_per_sample = _data.size() / _shape[0];
    Tensor result(new_shape);
    std::copy(_data.begin() + start * elements_per_sample,
              _data.begin() + end * elements_per_sample,
              result._data.begin());
    return result;
}

float Tensor::sum_squares() const {
    float sum = 0.0f;
    for (float val : _data) sum += val * val;
    return sum;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (_shape != other._shape) throw std::invalid_argument("Shape mismatch for subtraction");
    Tensor result(_shape);
    std::transform(_data.begin(), _data.end(), other._data.begin(),
                   result._data.begin(), std::minus<float>());
    return result;
}

Tensor operator-(float value, const Tensor& tensor) {
    Tensor result(tensor._shape);
    std::transform(tensor._data.begin(), tensor._data.end(),
                   result._data.begin(), [value](float x) { return value - x; });
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (_shape != other._shape) throw std::invalid_argument("Shape mismatch for addition");
    Tensor result(_shape);
    std::transform(_data.begin(), _data.end(), other._data.begin(),
                   result._data.begin(), std::plus<float>());
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (_shape != other._shape) throw std::invalid_argument("Shape mismatch for multiplication");
    Tensor result(_shape);
    std::transform(_data.begin(), _data.end(), other._data.begin(),
                   result._data.begin(), std::multiplies<float>());
    return result;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    Tensor result(tensor._shape);
    std::transform(tensor._data.begin(), tensor._data.end(),
                   result._data.begin(), [scalar](float x) { return scalar * x; });
    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    if (_shape.size() != 2 || other._shape.size() != 2 || _shape[1] != other._shape[0]) {
        throw std::invalid_argument("Invalid shapes for dot product");
    }
    Tensor result({_shape[0], other._shape[1]});
    for (size_t i = 0; i < _shape[0]; ++i) {
        for (size_t j = 0; j < other._shape[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < _shape[1]; ++k) {
                sum += (*this)({i, k}) * other({k, j});
            }
            result({i, j}) = sum;
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (_shape.size() != 2) throw std::invalid_argument("Transpose requires 2D tensor");
    Tensor result({_shape[1], _shape[0]});
    for (size_t i = 0; i < _shape[0]; ++i) {
        for (size_t j = 0; j < _shape[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        _shape = other._shape;
        _data = other._data;
    }
    return *this;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (new_size != _data.size()) {
        throw std::invalid_argument("New shape size must match total data size");
    }
    return Tensor(new_shape, _data); // Assuming a constructor Tensor(shape, data) exists
}

void Tensor::print() const {
    std::cout << "Tensor shape: (";
    for (size_t dim : _shape) std::cout << dim << ", ";
    std::cout << ")\n";
    for (size_t i = 0; i < _data.size(); ++i) {
        std::cout << _data[i] << " ";
        if (_shape.size() > 1 && (i + 1) % _shape[1] == 0) std::cout << "\n";
    }
    std::cout << "\n";
}
