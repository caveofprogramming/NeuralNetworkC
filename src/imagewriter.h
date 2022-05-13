#pragma once

#include <string>

#include "mnistloader.h"
#include "neuralnet.h"

namespace cave
{
    class ImageWriter
    {
    private:
        MNISTLoader loader_;

        int rows_{0};
        int cols_{0};
        NeuralNet neuralNet_;
        bool useNeuralNet_{false};

    public:
        ImageWriter(std::string inputDir): loader_(1000, inputDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        {
            std::string file = "saved.ann";
            try
            {
                neuralNet_.load(file);
                useNeuralNet_ = true;
                std::cout << "Loaded '" << file << "'" << std::endl;
            }
            catch(const std::exception& e)
            {
                std::cout << "No valid '" << file << "' found; not using neural net." << std::endl;
            }
        }

        bool write(std::string outputDir);

    private:
        void writeImages(Matrix &images, Matrix &labels, char *imageData);
        void writeLabels(Matrix &labels, std::ofstream &out);
    };
}