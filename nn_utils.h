#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class NNUtils {
public:
    // Генератор случайных чисел (синглтон)
    static std::mt19937& get_rng() {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        return rng;
    }

    // Инициализация весов (Xavier)
    static double xavier_init(int fan_in, int fan_out) {
        std::uniform_real_distribution<double> dist(-std::sqrt(6.0 / (fan_in + fan_out)), std::sqrt(6.0 / (fan_in + fan_out)));
        return dist(get_rng());
    }

    // Инициализация весов (He)
    static double he_init(int fan_in) {
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / fan_in));
        return dist(get_rng());
    }

    // Инициализация весов (случайная нормаль)
    static double random_normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(get_rng());
    }

    // Нормализация в диапазон [0, 1]
    static void normalize(std::vector<double>& data) {
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        for (double& x : data) {
            x = (x - min_val) / (max_val - min_val);
        }
    }

    // Стандартизация (Z-score)
    static void standardize(std::vector<double>& data) {
        double mean = 0.0, stddev = 0.0;
        for (double x : data) mean += x;
        mean /= data.size();

        for (double x : data) stddev += (x - mean) * (x - mean);
        stddev = std::sqrt(stddev / data.size());

        for (double& x : data) x = (x - mean) / stddev;
    }

    // Вычисление сигмоиды
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Вычисление ReLU
    static double relu(double x) {
        return (x > 0) ? x : 0;
    }

    // Вычисление производной ReLU
    static double relu_derivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }
};

#endif // NN_UTILS_H
