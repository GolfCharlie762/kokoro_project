#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "layers/layer.h"
#include "math/tensor.h"
#include <vector>

class Embedding : public Layer {
public:
    Embedding(size_t vocab_size, size_t embedding_dim);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;

    const Tensor& getWeights() const { return weights; }
    size_t getVocabSize() const { return vocab_size; }
    size_t getEmbeddingDim() const { return embedding_dim; }

    void setWeights(const Tensor& new_weights);

    void save(std::ofstream& file) const ;
    static std::unique_ptr<Embedding> load(std::ifstream& file);
    void print() const;

private:
    size_t vocab_size;     // Number of unique tokens/indices
    size_t embedding_dim;  // Dimension of embedding vectors
    Tensor weights;        // Embedding matrix: {vocab_size, embedding_dim}
    Tensor input_cache;    // Cached input indices
};

#endif // EMBEDDING_H
