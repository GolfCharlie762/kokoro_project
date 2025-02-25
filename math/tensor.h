#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>

class Tensor {
public:
    // Constructors
    Tensor() = default; // Default constructor
    Tensor(const std::vector<size_t>& shape); // Shape-only constructor
    Tensor(const std::vector<size_t>& shape, float value); // Shape with initial value
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data); // Shape with data

    // Accessors
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;
    float& at(size_t flat_index) { return _data[flat_index]; } // Flat index access
    const float& at(size_t flat_index) const { return _data[flat_index]; }

    // Shape and size
    const std::vector<size_t>& shape() const { return _shape; }
    size_t size() const { return _data.size(); }

    // Operations
    void fill(float value);
    void randomize(float min, float max);
    Tensor slice(size_t start, size_t end) const; // For batching
    float sum_squares() const; // For MSE loss
    Tensor operator-(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Element-wise
    Tensor dot(const Tensor& other) const;
    Tensor transpose() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor& operator=(const Tensor& other); // Assignment operator

    // Friend scalar operations
    friend Tensor operator-(float value, const Tensor& tensor);
    friend Tensor operator*(float scalar, const Tensor& tensor);

    // Data access
    float* data() { return _data.data(); }
    const float* data() const { return _data.data(); }

    // Debugging
    void print() const;

private:
    std::vector<size_t> _shape;
    std::vector<float> _data;
    size_t compute_index(const std::vector<size_t>& indices) const;
};

#endif // TENSOR_H
