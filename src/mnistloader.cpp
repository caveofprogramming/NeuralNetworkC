#include "mnistloader.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include "matrix.h"

namespace cave
{
    TrainingData MNISTLoader::load()
    {
        TrainingData trainingData;

        trainingData.input = loadImages();
        trainingData.expected = loadLabels();

        if(trainingData.input.size() != trainingData.expected.size())
        {
            std::stringstream ss;
            ss << "Image data contains " << trainingData.input.size();
            ss << " items but label data contains " << trainingData.expected.size();
            ss << " items.";
            throw std::logic_error(ss.str());
            return trainingData;
        }

        std::cout << "Loaded images and labels." << std::endl;

        return trainingData;
    }

    std::vector<Matrix> MNISTLoader::loadImages()
    {
        std::vector<Matrix> images;

        imageStream_.open(imageFile_, std::ios::binary);

        if (!imageStream_.is_open())
        {
            std::cerr << "Unable to open " << imageFile_ << std::endl;
            return images;
        }

        if (readInt(imageStream_) != 2051)
        {
            std::cerr << "Not an MNIST image file: " << labelFile_ << std::endl;
            return images;
        }

        int items = readInt(imageStream_);
        imageHeight_ = readInt(imageStream_);
        imageWidth_ = readInt(imageStream_);
        int inputSize = imageHeight_ * imageWidth_;
        int numberBatches = std::ceil(double(items) / batchSize_);

        int totalItemsRead = 0;

        int maxReadSize = batchSize_ * inputSize;

        auto imageData = std::make_unique<char[]>(maxReadSize);

        for (int i = 0; i < numberBatches; ++i)
        {
            int itemsToRead = std::min(batchSize_, items - totalItemsRead);

            int bytesToRead = itemsToRead * inputSize;

            imageStream_.read(imageData.get(), bytesToRead);

            if (!imageStream_)
            {
                std::cerr << "Unable to fully read " << imageFile_ << std::endl;
                return images;
            }
            cave::Matrix batch(inputSize, itemsToRead, [&](int row, int col, int index){
                int dataIndex = col * inputSize + row;
                uint8_t byte = imageData[dataIndex];
                return byte / 256.0;;
            });

            images.push_back(std::move(batch));

            totalItemsRead += itemsToRead;
        }

        imageStream_.close();

        return images;
    }

    std::vector<Matrix> MNISTLoader::loadLabels()
    {
        std::vector<Matrix> labels;

        std::cout << "Loading labels." << std::endl;

        labelStream_.open(labelFile_, std::ios::binary);

        if (!labelStream_.is_open())
        {
            std::cerr << "Unable to open " << labelFile_ << std::endl;
            return labels;
        }

        if (readInt(labelStream_) != 2049)
        {
            std::cerr << "Not an MNIST label file: " << labelFile_ << std::endl;
            return labels;
        }

        int items = readInt(labelStream_);
        int numberBatches = std::ceil(double(items) / batchSize_);
  
        int labelSize = 10;

        auto labelData = std::make_unique<char[]>(batchSize_);

        int totalItemsRead = 0;

        for (int i = 0; i < numberBatches; ++i)
        {
            int itemsToRead = std::min(batchSize_, items - totalItemsRead);

            int bytesToRead = itemsToRead;

            labelStream_.read(labelData.get(), bytesToRead);

            if (!imageStream_)
            {
                std::cerr << "Unable to fully read " << labelFile_ << std::endl;
                return labels;
            }

            cave::Matrix batch(labelSize, itemsToRead);

            for(int item = 0; item < itemsToRead; ++item)
            {
                int value = labelData[item];

                batch.set(value, item, 1);
            }

            labels.push_back(std::move(batch));

            totalItemsRead += itemsToRead;
        }

        labelStream_.close();

        return labels;
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
}