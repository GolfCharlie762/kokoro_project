#include "layers/preprocessing/text/text_vectorization.h"
#include <sstream>
#include <algorithm>
#include <fstream>

TextVectorizationLayer::TextVectorizationLayer(size_t max_tokens, size_t max_length, const std::string& delimiter)
    : max_tokens(max_tokens), max_length(max_length), delimiter(delimiter), vocab(), input_cache(Tensor()) {
    vocab[""] = 0; // Reserve 0 for padding/OOV
}

TextVectorizationLayer::TextVectorizationLayer(const std::vector<std::string>& vocabulary, size_t max_length, const std::string& delimiter)
    : max_tokens(vocabulary.size() + 1), max_length(max_length), delimiter(delimiter), vocab(), input_cache(Tensor()) {
    vocab[""] = 0; // Padding/OOV
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        vocab[vocabulary[i]] = i + 1;
    }
}

Tensor TextVectorizationLayer::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != 1) {
        throw std::invalid_argument("Input must be 2D with shape {batch_size, 1} containing text indices");
    }
    input_cache = input;
    size_t batch_size = input.shape()[0];
    Tensor output({batch_size, max_length});
    float* output_data = output.data();
    const float* input_data = input.data();

    for (size_t b = 0; b < batch_size; ++b) {
        std::string text = std::to_string(static_cast<int>(input_data[b])); // Assuming input is text index
        auto tokens = tokenize(text);
        size_t len = std::min(max_length, tokens.size());
        for (size_t t = 0; t < len; ++t) {
            auto it = vocab.find(tokens[t]);
            output_data[b * max_length + t] = (it != vocab.end()) ? static_cast<float>(it->second) : 0.0f;
        }
        // Remaining positions are already 0 due to Tensor constructor
    }

    return output;
}

Tensor TextVectorizationLayer::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.shape().size() != 2 || grad_output.shape()[0] != input_cache.shape()[0] ||
        grad_output.shape()[1] != max_length) {
        throw std::invalid_argument("Gradient shape must match output shape {batch_size, max_length}");
    }
    // No trainable parameters in this layer; gradient passes through unchanged to input shape
    Tensor grad_input(input_cache.shape());
    grad_input.fill(0.0f); // No gradient flows back to input text indices
    return grad_input;
}

void TextVectorizationLayer::adapt(const std::vector<std::string>& texts) {
    buildVocabulary(texts);
}

void TextVectorizationLayer::setVocabulary(const std::vector<std::string>& new_vocab) {
    vocab.clear();
    vocab[""] = 0; // Padding/OOV
    size_t limit = (max_tokens > 0) ? std::min(new_vocab.size(), max_tokens - 1) : new_vocab.size();
    for (size_t i = 0; i < limit; ++i) {
        vocab[new_vocab[i]] = i + 1;
    }
}

std::vector<std::string> TextVectorizationLayer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, delimiter[0])) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

void TextVectorizationLayer::buildVocabulary(const std::vector<std::string>& texts) {
    std::vector<std::pair<std::string, size_t>> token_counts;
    std::unordered_map<std::string, size_t> counts;
    for (const auto& text : texts) {
        auto tokens = tokenize(text);
        for (const auto& token : tokens) {
            counts[token]++;
        }
    }
    for (const auto& pair : counts) {
        token_counts.emplace_back(pair.first, pair.second);
    }
    std::sort(token_counts.begin(), token_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    vocab.clear();
    vocab[""] = 0; // Padding/OOV
    size_t limit = (max_tokens > 0) ? std::min(token_counts.size(), max_tokens - 1) : token_counts.size();
    for (size_t i = 0; i < limit; ++i) {
        vocab[token_counts[i].first] = i + 1;
    }
}

void TextVectorizationLayer::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&max_tokens), sizeof(max_tokens));
    file.write(reinterpret_cast<const char*>(&max_length), sizeof(max_length));
    size_t delim_size = delimiter.size();
    file.write(reinterpret_cast<const char*>(&delim_size), sizeof(delim_size));
    file.write(delimiter.c_str(), delim_size);
    size_t vocab_size = vocab.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    for (const auto& pair : vocab) {
        size_t key_size = pair.first.size();
        file.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        file.write(pair.first.c_str(), key_size);
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
    size_t shape_size = input_cache.shape().size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(input_cache.shape().data()), shape_size * sizeof(size_t));
    file.write(reinterpret_cast<const char*>(input_cache.data()), input_cache.size() * sizeof(float));
}

std::unique_ptr<TextVectorizationLayer> TextVectorizationLayer::load(std::ifstream& file) {
    size_t max_tokens, max_length;
    file.read(reinterpret_cast<char*>(&max_tokens), sizeof(max_tokens));
    file.read(reinterpret_cast<char*>(&max_length), sizeof(max_length));
    size_t delim_size;
    file.read(reinterpret_cast<char*>(&delim_size), sizeof(delim_size));
    std::string delimiter(delim_size, '\0');
    file.read(&delimiter[0], delim_size);
    auto layer = std::make_unique<TextVectorizationLayer>(max_tokens, max_length, delimiter);

    size_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    for (size_t i = 0; i < vocab_size; ++i) {
        size_t key_size;
        file.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, '\0');
        file.read(&key[0], key_size);
        size_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        layer->vocab[key] = value;
    }

    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));
    layer->input_cache = Tensor(shape);
    file.read(reinterpret_cast<char*>(layer->input_cache.data()), layer->input_cache.size() * sizeof(float));

    return layer;
}

void TextVectorizationLayer::print() const {
    std::cout << "TextVectorizationLayer\n";
    std::cout << "Max Tokens: " << max_tokens << "\n";
    std::cout << "Max Length: " << max_length << "\n";
    std::cout << "Delimiter: '" << delimiter << "'\n";
    std::cout << "Vocabulary Size: " << vocab.size() << "\n";
    std::cout << "Vocabulary (first 5 entries):\n";
    size_t count = 0;
    for (const auto& pair : vocab) {
        if (count >= 5) break;
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
        count++;
    }
    std::cout << "Input Cache:\n";
    input_cache.print();
}
