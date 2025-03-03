#ifndef TEXT_VECTORIZATION_H
#define TEXT_VECTORIZATION_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <string>
#include <vector>
#include <unordered_map>

class TextVectorizationLayer : public Layer {
public:
    TextVectorizationLayer(size_t max_tokens, size_t max_length, const std::string& delimiter = " ");
    TextVectorizationLayer(const std::vector<std::string>& vocabulary, size_t max_length, const std::string& delimiter = " ");

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    void adapt(const std::vector<std::string>& texts);
    size_t getMaxTokens() const { return max_tokens; }
    size_t getMaxLength() const { return max_length; }
    const std::unordered_map<std::string, size_t>& getVocabulary() const { return vocab; }
    void setVocabulary(const std::vector<std::string>& new_vocab);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<TextVectorizationLayer> load(std::ifstream& file);
    void print() const;

private:
    size_t max_tokens;                          // Maximum vocabulary size (0 = unlimited)
    size_t max_length;                          // Maximum sequence length
    std::string delimiter;                      // Token delimiter (e.g., space)
    std::unordered_map<std::string, size_t> vocab; // Token-to-index mapping
    Tensor input_cache;                         // Cached input for backward pass

    std::vector<std::string> tokenize(const std::string& text) const;
    void buildVocabulary(const std::vector<std::string>& texts);
};

#endif // TEXT_VECTORIZATION_H
