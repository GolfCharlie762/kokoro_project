#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>

class Tensor {
public:
    // Конструктор
    Tensor(const std::vector<size_t>& shape);

    // Доступ к элементам тензора по индексам
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;

    // Получить форму тензора
    std::vector<size_t> shape() const;

    // Получить общее количество элементов в тензоре
    size_t size() const;

    // Заполнить тензор значением
    void fill(float value);

    // Заполнить тензор случайными значениями
    void randomize(float min, float max);

    // Вывести тензор (для отладки)
    void print() const;

    // Перегрузка оператора вычитания (Tensor - Tensor)
    Tensor operator-(const Tensor& other) const;

    // Перегрузка оператора вычитания (int - Tensor)
    friend Tensor operator-(int value, const Tensor& tensor);

    // Перегрузка оператора +
    Tensor operator+(const Tensor& other) const;

    // Скалярное произведение
    Tensor dot(const Tensor& other) const;

    // Транспонирование
    Tensor transpose() const;

    // Перегрузка оператора умножения
    Tensor operator*(const Tensor& other) const;

private:
    std::vector<size_t> _shape; // Форма тензора (например, {2, 3} для матрицы 2x3)
    std::vector<float> _data;   // Данные тензора (хранятся в одномерном массиве)
};

#endif // TENSOR_H
