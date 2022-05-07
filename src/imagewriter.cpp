#include "imagewriter.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include "bitmapfileheader.h"
#include "bitmapinfoheader.h"

using namespace caveofprogramming;

namespace cave
{
    bool ImageWriter::write(std::string outputDir)
    {
        ImageMetaData &metaData = loader_.open();
        batchDataSize_ = metaData.batchSize * metaData.width * metaData.height * 3;

        std::cout << "Writing " << metaData.items << " items" << std::endl;

        cols_ = 100;
        rows_ = metaData.batchSize / cols_;

        assert(rows_ * cols_ == metaData.batchSize);

        imageWidth_ = metaData.width;
        imageHeight_ = metaData.height;

        BitmapFileHeader fileHeader;
        BitmapInfoHeader infoHeader;

        fileHeader.fileSize = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + batchDataSize_;
        fileHeader.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

        infoHeader.width = cols_ * metaData.width;
        infoHeader.height = rows_ * metaData.height;

        for (int batch = 0; batch < metaData.numberBatches; ++batch)
        {
            std::stringstream imageFilePath;
            imageFilePath << outputDir << "/image" << batch << ".bmp";

            std::stringstream labelFilePath;
            labelFilePath << outputDir << "/labels" << batch << ".txt";

            std::cout << "Writing " << imageFilePath.str() << std::endl;
            std::cout << "Writing " << labelFilePath.str() << std::endl;

            BatchData batchData = loader_.getBatch();

            std::ofstream imageFile;
            imageFile.open(imageFilePath.str(), ios::binary);

            if (!imageFile)
            {
                return false;
            }

            std::ofstream labelFile;
            labelFile.open(labelFilePath.str());

            if (!labelFile)
            {
                return false;
            }

            auto imageData = std::make_unique<char[]>(batchDataSize_);

            writeImages(batchData, imageData.get());
            writeLabels(batchData, labelFile);

            imageFile.write((char *)&fileHeader, sizeof(fileHeader));
            imageFile.write((char *)&infoHeader, sizeof(infoHeader));
            imageFile.write((char *)imageData.get(), batchDataSize_);

            imageFile.close();
            labelFile.close();
        }

        loader_.close();

        return true;
    }

    void ImageWriter::writeLabels(BatchData &batchData, std::ofstream &out)
    {
        for(int labelIndex = 0; labelIndex < batchData.numberRead; ++labelIndex)
        {
            int label = 0;

            for(int i = 0; i < 10; i++)
            {
                double value = batchData.expected[labelIndex * 10 + i];

                if(std::abs(value - 1) < 0.1)
                {
                    label = i;
                    break;
                }
            }

            if(labelIndex % cols_ == 0)
            {
                out << std::endl;
            }

            out << label;
        }
    }

    void ImageWriter::writeImages(BatchData &batchData, char *imageData)
    {
        auto pixelData = batchData.input;

        int imageSize = imageWidth_ * imageHeight_;
        int montageWidth = cols_ * imageWidth_;
        int montageHeight = rows_ * imageHeight_;

        for (int pixelIndex = 0; pixelIndex < batchDataSize_ / 3; ++pixelIndex)
        {
            double value = batchData.input[pixelIndex];

            int imageNumber = pixelIndex/imageSize;
            int pixelNumber = pixelIndex % imageSize;

            int montageRow = imageNumber / cols_;
            int montageCol = imageNumber % cols_;
            int pixelRow = pixelNumber / imageWidth_;
            int pixelCol = pixelNumber % imageWidth_;

            int x = montageCol * imageWidth_ + pixelCol;
            int y = montageRow * imageHeight_ + pixelRow;

            int destinationPixelIndex = ((montageHeight - 1) - y) * montageWidth + x;

            imageData[destinationPixelIndex * 3] = char(value * 255);
            imageData[destinationPixelIndex * 3 + 1] = char(value * 255);
            imageData[destinationPixelIndex * 3 + 2] = char(value * 255);
        }
    }

}