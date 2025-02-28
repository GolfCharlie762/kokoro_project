#include "quantizer/quantizer.h"
#include <algorithm>
#include <cmath>

///TEST_PROGRAM////
// Test data
//std::vector<float> data = {-1.5f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
//Tensor input_tensor({1, 6}, data);

//// Create quantizer
//Quantizer quantizer(8); // 8-bit quantization

//// Test 1: Quantize
//std::cout << "Original Tensor:\n";
//input_tensor.print();

//Tensor quantized = quantizer.quantize(input_tensor);
//std::cout << "Quantized Tensor (8-bit):\n";
//quantized.print();

//// Test 2: Dequantize
//Tensor dequantized = quantizer.dequantize(quantized);
//std::cout << "Dequantized Tensor:\n";
//dequantized.print();

//std::cout << "Scale: " << quantizer.getScale() << "\n";
//std::cout << "Zero Point: " << quantizer.getZeroPoint() << "\n";

//// Test 3: Uniform Quantization
//Tensor uniform_quantized = quantizer.uniformQuantize(input_tensor, -2.0f, 2.0f);
//std::cout << "Uniform Quantized Tensor (-2 to 2):\n";
//uniform_quantized.print();

//Tensor uniform_dequantized = quantizer.dequantize(uniform_quantized);
//std::cout << "Uniform Dequantized Tensor:\n";
//uniform_dequantized.print();

//// Test 4: Change Bit Width
//quantizer.setBitWidth(4); // Switch to 4-bit
//Tensor quantized_4bit = quantizer.quantize(input_tensor);
//std::cout << "Quantized Tensor (4-bit):\n";
//quantized_4bit.print();

//Tensor dequantized_4bit = quantizer.dequantize(quantized_4bit);
//std::cout << "Dequantized Tensor (4-bit):\n";
//dequantized_4bit.print();

//std::cout << "Scale (4-bit): " << quantizer.getScale() << "\n";
//std::cout << "Zero Point (4-bit): " << quantizer.getZeroPoint() << "\n";



Quantizer::Quantizer(size_t bit_width)
    : bit_width(bit_width), scale(0.0f), zero_point(0), min_val(0.0f), max_val(0.0f) {
    if (bit_width < 1 || bit_width > 32) {
        throw std::invalid_argument("Bit width must be between 1 and 32");
    }
}

Tensor Quantizer::quantize(const Tensor& data) {
    computeQuantizationParams(data);
    Tensor result(data.shape());
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t size = data.size();
    int32_t q_min = -(1 << (bit_width - 1)); // e.g., -128 for 8-bit
    int32_t q_max = (1 << (bit_width - 1)) - 1; // e.g., 127 for 8-bit

    for (size_t i = 0; i < size; ++i) {
        int32_t q_val = static_cast<int32_t>(std::round((data_ptr[i] - min_val) / scale)) + zero_point;
        q_val = std::max(q_min, std::min(q_max, q_val));
        result_data[i] = static_cast<float>(q_val); // Store as float for Tensor compatibility
    }

    return result;
}

Tensor Quantizer::dequantize(const Tensor& quantized_data) {
    Tensor result(quantized_data.shape());
    float* result_data = result.data();
    const float* q_data = quantized_data.data();
    size_t size = quantized_data.size();

    for (size_t i = 0; i < size; ++i) {
        result_data[i] = scale * (q_data[i] - zero_point) + min_val;
    }

    return result;
}

Tensor Quantizer::uniformQuantize(const Tensor& data, float min_val, float max_val) {
    this->min_val = min_val;
    this->max_val = max_val;
    float range = max_val - min_val;
    if (range <= 0) {
        throw std::invalid_argument("Max value must be greater than min value for uniform quantization");
    }
    int32_t levels = 1 << bit_width; // e.g., 256 for 8-bit
    scale = range / (levels - 1);
    zero_point = 0; // No offset for uniform quantization

    Tensor result(data.shape());
    float* result_data = result.data();
    const float* data_ptr = data.data();
    size_t size = data.size();
    int32_t q_max = levels - 1;

    for (size_t i = 0; i < size; ++i) {
        int32_t q_val = static_cast<int32_t>(std::round((data_ptr[i] - min_val) / scale));
        q_val = std::max(0, std::min(q_max, q_val));
        result_data[i] = static_cast<float>(q_val);
    }

    return result;
}

void Quantizer::setBitWidth(size_t new_bit_width) {
    if (new_bit_width < 1 || new_bit_width > 32) {
        throw std::invalid_argument("Bit width must be between 1 and 32");
    }
    bit_width = new_bit_width;
}

void Quantizer::computeQuantizationParams(const Tensor& data) {
    const float* data_ptr = data.data();
    size_t size = data.size();
    min_val = *std::min_element(data_ptr, data_ptr + size);
    max_val = *std::max_element(data_ptr, data_ptr + size);
    float range = max_val - min_val;
    if (range <= 0) {
        scale = 1.0f;
        zero_point = 0;
        return;
    }
    int32_t levels = 1 << bit_width; // e.g., 256 for 8-bit
    scale = range / (levels - 1);
    zero_point = static_cast<int32_t>(-min_val / scale);
}
