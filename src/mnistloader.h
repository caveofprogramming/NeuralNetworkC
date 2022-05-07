#pragma once

#include <string>
#include <fstream>
#include <cstdint>
#include <mutex>

#include "loader.h"

namespace cave
{
    struct ImageMetaData: public MetaData
    {
        int width;
        int height;
    };

    class MNISTLoader : public Loader
    {
    private:
        ImageMetaData metaData_;

        std::ifstream imageStream_;
        std::ifstream labelStream_;

        std::string imageFile_;
        std::string labelFile_;

        std::uint32_t readInt(std::ifstream &in);

        int totalItemsRead_{0};

        std::mutex mtxRead_;

    public:
        MNISTLoader(int batchSize, std::string inputDir, std::string imageFile, std::string labelFile) 
        {
            metaData_.batchSize = batchSize;
            imageFile_ = inputDir + "/" + imageFile;
            labelFile_ = inputDir + "/" + labelFile;
        } 

        ImageMetaData &open();
        ImageMetaData &getMetaData();
        BatchData getBatch();
        void close();
    };
}