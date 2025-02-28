#ifndef DATA_PREPROCESSOR_H
#define DATA_PREPROCESSOR_H

#include "math/tensor.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>

/**
 * @class DataPreprocessor
 * @brief Provides efficient data preprocessing utilities for neural networks.
 *
 * This class offers methods for numerical, text, and image data preprocessing, optimized for use with Tensor objects.
 *
 * **Numerical Preprocessing:**
 * - standardize: Centers data to zero mean and unit variance.
 * - normalize: Scales data to a specified range (default [0, 1]).
 * - clip: Limits values to a specified range.
 * - pad: Extends tensor to a target shape with padding (default 0.0).
 * - oneHotEncode: Converts indices to one-hot vectors.
 * - resize: Adjusts tensor size, truncating or padding as needed.
 *
 * **Text Preprocessing:**
 * - tokenizeText: Splits text into tokens using a delimiter (default space).
 * - textToTensor: Maps tokens to indices using a vocabulary.
 * - buildVocabulary: Creates a word-to-index map from text data.
 *
 * **Image Preprocessing:**
 * - grayscale: Converts RGB images to grayscale using luminance weights (0.299R, 0.587G, 0.114B).
 * - normalizeImage: Normalizes image pixel values with mean and std deviation.
 * - cropImage: Extracts a region from an image tensor.
 *
 * **Statistical Utilities:**
 * - mean: Computes the average of tensor elements.
 * - variance: Calculates sample variance given the mean.
 * - stddev: Computes sample standard deviation given the mean.
 *
 * Usage: Call static methods directly with Tensor objects or vectors as needed. All methods are optimized for raw pointer access.
 */

class DataPreprocessor {
public:
    static Tensor standardize(const Tensor& data);
    static Tensor normalize(const Tensor& data, float min_val = 0.0f, float max_val = 1.0f);
    static Tensor clip(const Tensor& data, float min_val, float max_val);
    static Tensor pad(const Tensor& data, const std::vector<size_t>& target_shape, float pad_value = 0.0f);
    static Tensor oneHotEncode(const Tensor& data, size_t num_classes);
    static Tensor resize(const Tensor& data, const std::vector<size_t>& new_shape);

    static std::vector<std::string> tokenizeText(const std::string& text, char delimiter = ' ');
    static Tensor textToTensor(const std::vector<std::string>& tokens, const std::map<std::string, size_t>& vocab);
    static std::map<std::string, size_t> buildVocabulary(const std::vector<std::string>& texts);

    static Tensor grayscale(const Tensor& image);
    static Tensor normalizeImage(const Tensor& image, float mean = 0.0f, float std = 255.0f);
    static Tensor cropImage(const Tensor& image, size_t start_h, size_t start_w, size_t height, size_t width);

    static float mean(const Tensor& data);
    static float variance(const Tensor& data, float mean_val);
    static float stddev(const Tensor& data, float mean_val);
};

#endif // DATA_PREPROCESSOR_H
