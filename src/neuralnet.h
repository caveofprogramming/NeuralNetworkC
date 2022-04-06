#pragma once

#include <vector>
#include <iostream>
#include <string>
#include "matrix.h"
#include "loader.h"
#include "transforms.h"

namespace cave
{

    class NeuralNet
    {
    private:
        enum Transform
        {
            DENSE = 0,
            RELU = 1,
            SOFTMAX = 2,
        };

        std::vector<std::string> transformNames{"DENSE", "RELU", "SOFTMAX"};

        std::vector<Matrix> weights_;
        std::vector<Matrix> biases_;

        std::vector<Transform> transforms_;

        double scaleInitialWeights_ = 1.0;

    private:
        void runForwards(MetaData &metaData, BatchData &batchData);

    public:
        NeuralNet(std::vector<int> layerSizes);
        void setScaleInitialWeights(double scale) { scaleInitialWeights_ = scale; };
        void fit(Loader trainingLoader);

        friend std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet);
    };
}