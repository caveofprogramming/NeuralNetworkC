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
        TrainingData data = loader_.load();

        int batchSize = data.input[0].cols();

        cols_ = 100;
        rows_ = batchSize / cols_;

        assert(rows_ * cols_ == batchSize);

        int imageWidth = loader_.getImageWidth();
        int imageHeight = loader_.getImageHeight();

        BitmapFileHeader fileHeader;
        BitmapInfoHeader infoHeader;

        int batchDataSize = batchSize * data.input[0].rows() * 3;

        fileHeader.fileSize = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + batchDataSize;
        fileHeader.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

        infoHeader.width = cols_ * imageWidth;
        infoHeader.height = rows_ * imageHeight;

        for (int batch = 0; batch < data.input.size(); ++batch)
        {
            std::stringstream imageFilePath;
            imageFilePath << outputDir << "/image" << batch << ".bmp";

            std::stringstream labelFilePath;
            labelFilePath << outputDir << "/labels" << batch << ".txt";

            std::cout << "Writing " << imageFilePath.str() << std::endl;
            std::cout << "Writing " << labelFilePath.str() << std::endl;

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

            auto imageData = std::make_unique<char[]>(batchDataSize);

            writeImages(data.input[batch], imageData.get());
            writeLabels(data.expected[batch], labelFile);

            imageFile.write((char *)&fileHeader, sizeof(fileHeader));
            imageFile.write((char *)&infoHeader, sizeof(infoHeader));
            imageFile.write((char *)imageData.get(), batchDataSize);
            
            imageFile.close();
            labelFile.close();
        }

        return true;
    }

    void ImageWriter::writeLabels(Matrix &labels, std::ofstream &out)
    {
        for (int labelIndex = 0; labelIndex < labels.cols(); ++labelIndex)
        {
            int label = 0;

            for (int i = 0; i < 10; i++)
            {
                double value = labels.get(i, labelIndex);

                if (std::abs(value - 1) < 0.1)
                {
                    label = i;
                    break;
                }
            }

            if (labelIndex % cols_ == 0)
            {
                out << std::endl;
            }

            out << label;
        }
    }

    void ImageWriter::writeImages(Matrix &images, char *imageData)
    {
        int imageWidth = loader_.getImageWidth();
        int imageHeight = loader_.getImageHeight();
        int montageWidth = cols_ * imageWidth;
        int montageHeight = rows_ * imageHeight;

        for (int imageIndex = 0; imageIndex < images.cols(); ++imageIndex)
        {
            int montageRow = imageIndex / cols_;
            int montageCol = imageIndex % cols_;

            for (int pixelIndex = 0; pixelIndex < images.rows(); ++pixelIndex)
            {
                int pixelRow = pixelIndex / imageWidth;
                int pixelCol = pixelIndex % imageWidth;

                int x = montageCol * imageWidth + pixelCol;
                int y = montageRow * imageHeight + pixelRow;

                int destinationPixelIndex = ((montageHeight - 1) - y) * montageWidth + x;

                double value = images.get(pixelIndex, imageIndex);

                imageData[destinationPixelIndex * 3] = char(value * 255);
                imageData[destinationPixelIndex * 3 + 1] = char(value * 255);
                imageData[destinationPixelIndex * 3 + 2] = char(value * 255);
            }
        }
    }

}