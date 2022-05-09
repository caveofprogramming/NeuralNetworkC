#pragma once

#include <string>
#include <fstream>
#include <cstdint>
#include <mutex>

#include "loader.h"

namespace cave
{
    class MNISTLoader : public Loader
    {
    private:
        std::ifstream imageStream_;
        std::ifstream labelStream_;

        std::string imageFile_;
        std::string labelFile_;

        std::uint32_t readInt(std::ifstream &in);

        std::vector<Matrix> loadImages();
        std::vector<Matrix> loadLabels();

        int batchSize_{0};
        int items_{0};
        int imageWidth_{0};
        int imageHeight_{0};

    public:
        MNISTLoader(int batchSize, std::string inputDir, std::string imageFile, std::string labelFile): batchSize_{batchSize}
        {
            imageFile_ = inputDir + "/" + imageFile;
            labelFile_ = inputDir + "/" + labelFile;
        } 

        int getImageWidth() { return imageWidth_; }
        int getImageHeight() { return imageHeight_; }

        TrainingData load();
    };
}