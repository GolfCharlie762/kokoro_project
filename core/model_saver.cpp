#include "model_saver.h"
#include "layers/dense_layer.h"
#include "layers/lstm.h"
#include "activations/relu.h"
#include "layers/dropout.h"
#include "layers/conv2d.h"
#include <stdexcept>
#include <sstream>

// Сохранить модель в файл
void ModelSaver::saveModel(const Model& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving model.");
    }

    // Сохраняем количество слоев
    size_t num_layers = model.getLayers().size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // Сохраняем каждый слой
    for (const auto& layer : model.getLayers()) {
        // Сохраняем тип слоя
        if (dynamic_cast<DenseLayer*>(layer.get())) {
            std::string layer_type = "DenseLayer";
            size_t type_size = layer_type.size();
            file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
            file.write(layer_type.c_str(), type_size);

            auto dense_layer = std::dynamic_pointer_cast<DenseLayer>(layer);

            // Получаем размеры
            size_t input_size = dense_layer->getInputSize();
            size_t output_size = dense_layer->getOutputSize();

            // Логирование размеров
            std::cout << "Saving DenseLayer with input_size: " << input_size
                      << ", output_size: " << output_size << "\n";

            // Записываем размеры
            file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
            file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

            // Получаем веса и смещения
            const Tensor& weights = dense_layer->getWeights();
            const Tensor& biases = dense_layer->getBiases();

            // Логирование размеров весов и смещений
            std::cout << "Weights shape: (";
            for (size_t dim : weights.shape()) std::cout << dim << ", ";
            std::cout << ")\n";
            std::cout << "Biases shape: (";
            for (size_t dim : biases.shape()) std::cout << dim << ", ";
            std::cout << ")\n";

            // Записываем веса и смещения
            file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
        }

        else if (dynamic_cast<ReLU*>(layer.get())) {
            std::string layer_type = "ReLU";
            size_t type_size = layer_type.size();
            file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
            file.write(layer_type.c_str(), type_size);
        } else if (dynamic_cast<Dropout*>(layer.get())) {
            std::string layer_type = "Dropout";
            size_t type_size = layer_type.size();
            file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
            file.write(layer_type.c_str(), type_size);

            // Сохраняем параметр rate
            auto dropout_layer = std::dynamic_pointer_cast<Dropout>(layer);
            float rate = dropout_layer->getRate();
            file.write(reinterpret_cast<const char*>(&rate), sizeof(rate));
        }
        // В секции сохранения слоев добавьте:
        else if (dynamic_cast<LSTM*>(layer.get())) {
            std::string layer_type = "LSTM";
            size_t type_size = layer_type.size();
            file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
            file.write(layer_type.c_str(), type_size);

            auto lstm_layer = std::dynamic_pointer_cast<LSTM>(layer);

            // Сохраняем размеры
            size_t input_size = lstm_layer->getInputSize();
            size_t hidden_size = lstm_layer->getHiddenSize();
            file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
            file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));

            // Сохраняем веса и смещения
            const Tensor& Wf = lstm_layer->getWf();
            const Tensor& Wi = lstm_layer->getWi();
            const Tensor& Wo = lstm_layer->getWo();
            const Tensor& Wc = lstm_layer->getWc();
            const Tensor& bf = lstm_layer->getBf();
            const Tensor& bi = lstm_layer->getBi();
            const Tensor& bo = lstm_layer->getBo();
            const Tensor& bc = lstm_layer->getBc();

            // Записываем данные
            file.write(reinterpret_cast<const char*>(Wf.data()), Wf.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(Wi.data()), Wi.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(Wo.data()), Wo.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(Wc.data()), Wc.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bf.data()), bf.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bi.data()), bi.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bo.data()), bo.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(bc.data()), bc.size() * sizeof(float));
        }
        else if (dynamic_cast<Conv2D*>(layer.get())) {
            std::string layer_type = "Conv2D";
            size_t type_size = layer_type.size();
            file.write(reinterpret_cast<const char*>(&type_size), sizeof(type_size));
            file.write(layer_type.c_str(), type_size);

            auto conv_layer = std::dynamic_pointer_cast<Conv2D>(layer);

            // Сохраняем метаданные
            size_t meta[5] = {
                conv_layer->getInputChannels(),
                conv_layer->getOutputChannels(),
                conv_layer->getKernelSize(),
                conv_layer->getStride(),
                conv_layer->getPadding()
            };
            file.write(reinterpret_cast<const char*>(meta), sizeof(meta));

            // Сохраняем веса и смещения
            const Tensor& kernels = conv_layer->getKernels();
            const Tensor& biases = conv_layer->getBiases();
            file.write(reinterpret_cast<const char*>(kernels.data()), kernels.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
        }
        else {
            throw std::runtime_error("Unsupported layer type for saving.");
        }
    }

    file.close();
}

// Загрузить модель из файла
void ModelSaver::loadModel(Model& model, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading model.");
    }

    // Загружаем количество слоев
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    // Загружаем каждый слой
    for (size_t i = 0; i < num_layers; ++i) {
        // Загружаем тип слоя
        size_t type_size;
        file.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
        std::string layer_type(type_size, '\0');
        file.read(&layer_type[0], type_size);
        if (layer_type == "DenseLayer") {
            // Читаем размеры
            size_t input_size, output_size;
            file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
            file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

            // Проверка на ошибки чтения
            if (file.fail()) {
                throw std::runtime_error("Failed to read DenseLayer parameters.");
            }

            // Проверка на нулевые или слишком большие размеры
            if (input_size == 0 || output_size == 0) {
                throw std::invalid_argument("DenseLayer input_size and output_size must be positive.");
            }
            if (input_size > 1000000 || output_size > 1000000) {
                throw std::invalid_argument("DenseLayer input_size or output_size is too large.");
            }

            // Логирование размеров
            std::cout << "Loading DenseLayer with input_size: " << input_size
                      << ", output_size: " << output_size << "\n";

            // Создаем тензоры для весов и смещений
            std::vector<size_t> weights_shape = {input_size, output_size};
            Tensor weights(weights_shape);
            Tensor biases({output_size});

            // Логирование размеров весов и смещений
            std::cout << "Weights shape: (";
            for (size_t dim : weights_shape) std::cout << dim << ", ";
            std::cout << ")\n";
            std::cout << "Biases shape: (" << output_size << ")\n";

            // Читаем веса и смещения
            file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));

            // Проверка на ошибки чтения
            if (file.fail()) {
                throw std::runtime_error("Failed to read DenseLayer weights or biases.");
            }

            // Создаем и добавляем слой
            auto dense_layer = std::make_shared<DenseLayer>(input_size, output_size);
            dense_layer->setWeights(weights);
            dense_layer->setBiases(biases);
            model.addLayer(dense_layer);
        }

         else if (layer_type == "ReLU") {
            // Добавляем слой ReLU
            model.addLayer(std::make_shared<ReLU>());
        } else if (layer_type == "Dropout") {
            // Загружаем параметр rate
            float rate;
            file.read(reinterpret_cast<char*>(&rate), sizeof(rate));

            // Создаем и добавляем слой Dropout
            auto dropout_layer = std::make_shared<Dropout>(rate);
            model.addLayer(dropout_layer);
        }else if (layer_type == "LSTM") {
            // Читаем размеры
            size_t input_size, hidden_size;
            file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
            file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));

            // Создаем тензоры для параметров
            Tensor Wf({input_size + hidden_size, hidden_size});
            Tensor Wi({input_size + hidden_size, hidden_size});
            Tensor Wo({input_size + hidden_size, hidden_size});
            Tensor Wc({input_size + hidden_size, hidden_size});
            Tensor bf({hidden_size}), bi({hidden_size}), bo({hidden_size}), bc({hidden_size});

            // Читаем данные
            file.read(reinterpret_cast<char*>(Wf.data()), Wf.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(Wi.data()), Wi.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(Wo.data()), Wo.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(Wc.data()), Wc.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bf.data()), bf.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bi.data()), bi.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bo.data()), bo.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(bc.data()), bc.size() * sizeof(float));

            // Создаем слой и устанавливаем параметры
            auto lstm_layer = std::make_shared<LSTM>(input_size, hidden_size);
            lstm_layer->setWeights(Wf, Wi, Wo, Wc, bf, bi, bo, bc);
            model.addLayer(lstm_layer);
        }
        else if (layer_type == "Conv2D") {
            // Читаем метаданные
            size_t meta[5];
            file.read(reinterpret_cast<char*>(meta), sizeof(meta));
            size_t input_channels = meta[0];
            size_t output_channels = meta[1];
            size_t kernel_size = meta[2];
            size_t stride = meta[3];
            size_t padding = meta[4];

            // Создаем слой
            auto conv_layer = std::make_shared<Conv2D>(input_channels, output_channels, kernel_size, stride, padding);

            // Читаем веса и смещения
            std::vector<size_t> kernels_shape = {output_channels, input_channels, kernel_size, kernel_size};
            Tensor kernels(kernels_shape);
            Tensor biases({output_channels});

            file.read(reinterpret_cast<char*>(kernels.data()), kernels.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));

            // Устанавливаем параметры
            conv_layer->setWeights(kernels, biases);
            model.addLayer(conv_layer);
        }
                else {
            throw std::runtime_error("Unsupported layer type for loading.");
        }
    }

    file.close();
}

// Вывести информацию о модели из файла
std::string ModelSaver::modelInfo(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading model info.");
    }

    std::stringstream info;
    info << "Model structure:\n";

    // Читаем количество слоев
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    info << "Layers: " << num_layers << "\n\n";

    for (size_t i = 0; i < num_layers; ++i) {
        info << "Layer " << (i + 1) << ":\n";

        // Читаем тип слоя
        size_t type_size;
        file.read(reinterpret_cast<char*>(&type_size), sizeof(type_size));
        std::string layer_type(type_size, '\0');
        file.read(&layer_type[0], type_size);
        info << "  Type: " << layer_type << "\n";

        // Обрабатываем данные слоя
        if (layer_type == "DenseLayer") {
            size_t input_size, output_size;
            file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
            file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
            info << "  Input size: " << input_size << "\n";
            info << "  Output size: " << output_size << "\n";

            // Пропускаем веса и смещения
            size_t weights_size = input_size * output_size;
            size_t biases_size = output_size;
            file.seekg((weights_size + biases_size) * sizeof(float), std::ios::cur);

        } else if (layer_type == "ReLU") {
            // Нет параметров
            info << "  Activation: ReLU\n";

        } else if (layer_type == "Dropout") {
            float rate;
            file.read(reinterpret_cast<char*>(&rate), sizeof(rate));
            info << "  Dropout rate: " << rate << "\n";

        } else if (layer_type == "Conv2D") {
            size_t meta[5];
            file.read(reinterpret_cast<char*>(meta), sizeof(meta));
            info << "  Input channels: " << meta[0] << "\n";
            info << "  Output channels: " << meta[1] << "\n";
            info << "  Kernel size: " << meta[2] << "\n";
            info << "  Stride: " << meta[3] << "\n";
            info << "  Padding: " << meta[4] << "\n";

            // Пропускаем ядра и смещения
            size_t kernels_size = meta[1] * meta[0] * meta[2] * meta[2];
            size_t biases_size = meta[1];
            file.seekg((kernels_size + biases_size) * sizeof(float), std::ios::cur);

        } else if (layer_type == "LSTM") {
            size_t input_size, hidden_size;
            file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
            file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
            info << "  Input size: " << input_size << "\n";
            info << "  Hidden size: " << hidden_size << "\n";

            // Пропускаем веса и смещения
            size_t params_size = 4 * (input_size + hidden_size) * hidden_size + 4 * hidden_size;
            file.seekg(params_size * sizeof(float), std::ios::cur);

        } else {
            throw std::runtime_error("Unsupported layer type: " + layer_type);
        }

        info << "\n";
    }

    file.close();
    return info.str();
}
