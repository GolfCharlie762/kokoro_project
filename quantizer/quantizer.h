#ifndef QUANTIZER_H
#define QUANTIZER_H

#include "math/tensor.h"
#include <vector>
#include <cstdint>
#include <limits>

/**
 * @class Quantizer
 * @brief Efficiently quantizes and dequantizes Tensor data for reduced precision.
 *
 * Provides methods to convert floating-point tensors to integer representations (e.g., int8, int16) and back,
 * optimizing storage and computation for neural networks.
 *
 * Key Features:
 * - quantize: Converts float tensor to quantized integers based on data range and bit width.
 * - dequantize: Reverts quantized integers to approximate float values.
 * - uniformQuantize: Quantizes with a user-specified range.
 * - Adjustable bit width (1–32 bits) for flexible precision control.
 *
 * Usage: Create with desired bit width, then call quantize/dequantize on Tensor objects.
 */

class Quantizer {
public:
    Quantizer(size_t bit_width = 8); // Default to 8-bit quantization
    Tensor quantize(const Tensor& data); // Quantize float to integer
    Tensor dequantize(const Tensor& quantized_data); // Dequantize integer back to float
    Tensor uniformQuantize(const Tensor& data, float min_val, float max_val); // Uniform quantization with custom range

    size_t getBitWidth() const { return bit_width; }
    float getScale() const { return scale; }
    int32_t getZeroPoint() const { return zero_point; }

    void setBitWidth(size_t new_bit_width);

private:
    size_t bit_width; // Number of bits (e.g., 8 for int8, 16 for int16)
    float scale;      // Scaling factor for quantization
    int32_t zero_point; // Zero point for quantization
    float min_val;    // Minimum value in original data
    float max_val;    // Maximum value in original data

    void computeQuantizationParams(const Tensor& data);
};

#endif // QUANTIZER_H
