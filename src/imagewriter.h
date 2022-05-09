#pragma once

#include <string>

#include "mnistloader.h"

namespace cave
{
    class ImageWriter
    {
    private:
        MNISTLoader loader_;

        int rows_{0};
        int cols_{0};

    public:
        ImageWriter(std::string inputDir): loader_(10000, inputDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        {
        }

        bool write(std::string outputDir);

    private:
        void writeImages(Matrix &images, char *imageData);
        void writeLabels(Matrix &labels, std::ofstream &out);
    };
}