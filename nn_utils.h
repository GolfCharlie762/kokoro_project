#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

class NNUtils {
public:
    static std::mt19937& get_rng() {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        return rng;
    }

    static float xavier_init(int fan_in, int fan_out) {
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(get_rng());
    }

    static float he_init(int fan_in) {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
        return dist(get_rng());
    }

    static float random_normal(float mean, float stddev) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(get_rng());
    }

    static float random_uniform(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(get_rng());
    }

    static void normalize(std::vector<float>& data, float min_val = 0.0f, float max_val = 1.0f) {
        if (data.empty()) return;
        float data_min = *std::min_element(data.begin(), data.end());
        float data_max = *std::max_element(data.begin(), data.end());
        if (data_max == data_min) return;
        float scale = (max_val - min_val) / (data_max - data_min);
        for (float& x : data) {
            x = min_val + (x - data_min) * scale;
        }
    }

    static void standardize(std::vector<float>& data) {
        if (data.empty()) return;
        float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
        float variance = 0.0f;
        for (float x : data) {
            float diff = x - mean;
            variance += diff * diff;
        }
        variance /= data.size();
        float stddev = std::sqrt(variance);
        if (stddev == 0.0f) return;
        for (float& x : data) {
            x = (x - mean) / stddev;
        }
    }

    static void clip(std::vector<float>& data, float min_val, float max_val) {
        for (float& x : data) {
            x = std::max(min_val, std::min(max_val, x));
        }
    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float sigmoid_derivative(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }

    static float relu(float x) {
        return std::max(0.0f, x);
    }

    static float relu_derivative(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    static float tanh(float x) {
        return std::tanh(x);
    }

    static float tanh_derivative(float x) {
        float t = std::tanh(x);
        return 1.0f - t * t;
    }

    static float leaky_relu(float x, float alpha = 0.01f) {
        return x > 0.0f ? x : alpha * x;
    }

    static float leaky_relu_derivative(float x, float alpha = 0.01f) {
        return x > 0.0f ? 1.0f : alpha;
    }

    static float softmax(std::vector<float>& logits, size_t index) {
        float sum_exp = 0.0f;
        for (float x : logits) {
            sum_exp += std::exp(x - *std::max_element(logits.begin(), logits.end()));
        }
        return std::exp(logits[index] - *std::max_element(logits.begin(), logits.end())) / sum_exp;
    }

    static float mean(const std::vector<float>& data) {
        if (data.empty()) return 0.0f;
        return std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    }

    static float variance(const std::vector<float>& data) {
        if (data.size() <= 1) return 0.0f;
        float m = mean(data);
        float sum_sq = 0.0f;
        for (float x : data) {
            float diff = x - m;
            sum_sq += diff * diff;
        }
        return sum_sq / (data.size() - 1);
    }

    static float stddev(const std::vector<float>& data) {
        return std::sqrt(variance(data));
    }

    static std::vector<size_t> argmax(const std::vector<float>& data) {
        std::vector<size_t> indices;
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                indices.clear();
                indices.push_back(i);
            } else if (data[i] == max_val) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    static void shuffle(std::vector<float>& data) {
        std::shuffle(data.begin(), data.end(), get_rng());
    }
};

#endif // NN_UTILS_H
