#include "mnistloader.h"

#include <iostream>
#include <cmath>

namespace cave
{
    ImageMetaData &MNISTLoader::open()
    {
        imageStream_.open(imageFile_, std::ios::binary);

        if (!imageStream_.is_open())
        {
            std::cerr << "Unable to open " << imageFile_ << std::endl;
            return metaData_;
        }

        labelStream_.open(labelFile_, std::ios::binary);

        if (!labelStream_.is_open())
        {
            std::cerr << "Unable to open " << labelFile_ << std::endl;
            return metaData_;
        }

        if (readInt(imageStream_) != 2051)
        {
            std::cerr << "Not an MNIST image file: " << labelFile_ << std::endl;
            return metaData_;
        }

        if (readInt(labelStream_) != 2049)
        {
            std::cerr << "Not an MNIST label file: " << labelFile_ << std::endl;
            return metaData_;
        }

        metaData_.items = readInt(labelStream_);

        if (metaData_.items != readInt(imageStream_))
        {
            std::cerr << "Numbers of items in image and label files differ" << std::endl;
            return metaData_;
        }

        metaData_.height = readInt(imageStream_);
        metaData_.width = readInt(imageStream_);
        metaData_.inputSize = metaData_.height * metaData_.width;

        metaData_.numberBatches = std::ceil(double(metaData_.items) / metaData_.batchSize);

        return metaData_;
    }

    std::uint32_t MNISTLoader::readInt(std::ifstream &in)
    {
        char bytes[4];
        uint32_t result = 0;
        in.read(bytes, 4);

        char *pInt = reinterpret_cast<char *>(&result);

        pInt[3] = bytes[0];
        pInt[2] = bytes[1];
        pInt[1] = bytes[2];
        pInt[0] = bytes[3];

        return result;
    }

    ImageMetaData &MNISTLoader::getMetaData()
    {
        return metaData_;
    }

    BatchData MNISTLoader::getBatch()
    {
        std::lock_guard<std::mutex> guard(mtxRead_);

        BatchData batchData;

        batchData.numberRead = std::min(metaData_.batchSize, metaData_.items - totalItemsRead_);
        totalItemsRead_ += batchData.numberRead;

        int sizeImageData = metaData_.width * metaData_.height * batchData.numberRead;
        int sizeLabelData = batchData.numberRead;

        auto imageData = std::make_unique<char[]>(sizeImageData);
        auto labelData = std::make_unique<char[]>(sizeLabelData);

        imageStream_.read(imageData.get(), sizeImageData);
        labelStream_.read(labelData.get(), sizeLabelData);

        batchData.input.resize(sizeImageData);
        batchData.expected.resize(sizeLabelData * 10);

        for(int i = 0; i < sizeImageData; ++i)
        {
            uint8_t byte = imageData[i];

            batchData.input[i] = byte/256.0;
        }

        for(int i = 0; i < sizeLabelData; ++i)
        {
            batchData.expected[i * 10 + labelData[i]] = 1;
        }

        return batchData;
    }

    void MNISTLoader::close()
    {
        imageStream_.close();
        labelStream_.close();
    }
}