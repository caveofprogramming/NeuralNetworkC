#include "neuralnet.h"

#include <random>
#include <iostream>

namespace cave
{
    std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet)
    {
        out << "Layers:\n\n";

        int weightIndex = 0;

        for (NeuralNet::Transform transform : neuralNet.transforms_)
        {
            out << neuralNet.transformNames[transform];

            if (transform == NeuralNet::DENSE)
            {
                Matrix weight = neuralNet.weights_[weightIndex++];
                out << " " << weight.rows() << " x " << weight.cols();
            }

            out << std::endl;
        }

        return out;
    }

    void NeuralNet::fit(Loader trainingLoader)
    {
        MetaData metaData = trainingLoader.open();

        for (int i = 0; i < metaData.numberBatches; ++i)
        {
            BatchData batchData = trainingLoader.getBatch();

            runForwards(metaData, batchData);
        }
    }

    void NeuralNet::runForwards(MetaData &metaData, BatchData &batchData)
    {
        Matrix input(metaData.inputSize, batchData.batchItemsRead, batchData.input);
        Matrix expected(metaData.outputSize, batchData.batchItemsRead, batchData.expected);

        for (int i = 0; i < weights_.size(); ++i)
        {
            Matrix weight = weights_[i];
            Matrix bias = biases_[i];
        }
    }

    NeuralNet::NeuralNet(std::vector<int> layerSizes)
    {
        std::default_random_engine generator;
        std::random_device rd;
        generator.seed(rd());

        std::normal_distribution<double> normal(-1, 1);

        int inputSize = layerSizes[0];

        for (int i = 1; i < layerSizes.size(); ++i)
        {
            int layerSize = layerSizes[i];

            Matrix weight(layerSize, inputSize, [&]()
                          { return scaleInitialWeights_ * normal(generator); });

            Matrix bias(layerSize, 1);

            weights_.push_back(weight);
            biases_.push_back(bias);
            transforms_.push_back(DENSE);
            transforms_.push_back(RELU);

            inputSize = layerSize;
        }

        transforms_[transforms_.size() - 1] = SOFTMAX;
    }
}