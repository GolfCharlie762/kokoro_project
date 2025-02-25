#ifndef MODEL_SAVER_H
#define MODEL_SAVER_H

#include "model.h"
#include <fstream>
#include <string>

class ModelSaver {
public:
    // Сохранить модель в файл
    static void saveModel(const Model& model, const std::string& filename);

    // Загрузить модель из файла
    static void loadModel(Model& model, const std::string& filename);

    //Получить информацию о модели из файла (по пути)
    std::string modelInfo(const std::string& filename);
};

#endif // MODEL_SAVER_H
