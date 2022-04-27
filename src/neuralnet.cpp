#include "neuralnet.h"

#include <random>
#include <iostream>
#include <exception>
#include <mutex>

#include "matrixfunctions.h"

namespace cave
{
    std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet)
    {
        if (neuralNet.weights_.size() > 0)
        {
            out << "Layers:\n\n";
        }

        int weightIndex = 0;

        for (NeuralNet::Transform transform : neuralNet.transforms_)
        {
            out << neuralNet.transformNames_[transform];

            if (transform == NeuralNet::DENSE)
            {
                Matrix weight = neuralNet.weights_[weightIndex++];
                out << " " << weight.rows() << " x " << weight.cols();
            }

            out << std::endl;
        }

        return out;
    }

    void NeuralNet::add(NeuralNet::Transform transform, int rows, int cols)
    {
        if (transform == DENSE)
        {
            std::default_random_engine generator;
            std::random_device rd;
            generator.seed(rd());
            std::normal_distribution<double> normal(-1, 1);

            if (cols == 0)
            {
                if (weights_.size() == 0)
                {
                    throw std::invalid_argument("Cols parameter must be supplied for first dense layer");
                }

                cols = weights_.back().rows();
            }

            Matrix weight(rows, cols, [&]()
                          { return scaleInitialWeights_ * normal(generator); });

            Matrix bias(rows, 1);

            weights_.push_back(weight);
            biases_.push_back(bias);
        }

        transforms_.push_back(transform);
    }

    Matrix NeuralNet::loss(BatchResult &result, Matrix &expected)
    {
        Matrix &output = result.io.back();
        return crossEntropy(output, expected);
    }

    Matrix NeuralNet::predict(Matrix &input)
    {
        if (weights_.size() > 0)
        {
            assert(input.rows() == weights_[0].cols());
        }

        BatchResult result = runForwards(input);

        return result.io.back();
    }

    std::vector<double> NeuralNet::predict(std::vector<double> input)
    {
        if (weights_.size() > 0)
        {
            assert(input.size() == weights_[0].cols());
        }

        Matrix mInput(input.size(), 1, input);
        BatchResult result = runForwards(mInput);

        return result.io.back().get();
    }

    void NeuralNet::fit(Loader &trainingLoader)
    {
        MetaData metaData = trainingLoader.open();

        if (weights_.size() > 0)
        {
            assert(metaData.inputSize == weights_[0].cols());
        }

        for (int i = 0; i < metaData.numberBatches; ++i)
        {
            BatchData batchData = trainingLoader.getBatch();
            Matrix input(metaData.inputSize, batchData.batchItemsRead, batchData.input);
            Matrix expected(metaData.outputSize, batchData.batchItemsRead, batchData.expected);

            BatchResult result = runForwards(input);
        }
    }

    void NeuralNet::fit(Matrix &input, Matrix &expected)
    {
        BatchResult batchResult = runForwards(input);
        batchResult = runBackwards(batchResult, expected);
        adjust(batchResult);
    }

    BatchResult NeuralNet::runForwards(Matrix input)
    {
        int weightIndex = 0;

        BatchResult result;

        result.io.push_back(input);

        for (Transform transform : transforms_)
        {
            switch (transform)
            {
            case DENSE:
            {
                result.weightInputs.push_back(input);

                Matrix &weight = weights_[weightIndex];
                Matrix &bias = biases_[weightIndex];

                input = (weight * input).modify([&](int row, int col, int index, double value)
                                                { return value + bias.get(row); });

                ++weightIndex;
            }
            break;
            case RELU:
                input = relu(input);
                break;
            case SOFTMAX:
                input = softmax(input);
                break;
            }

            result.io.push_back(input);
        }

        return result;
    }

    BatchResult NeuralNet::runBackwards(BatchResult batchResult, Matrix expected)
    {
        auto io = batchResult.io;

        Matrix output = io.back();

        if(transforms_.back() != SOFTMAX)
        {
            throw std::logic_error("Final transform must be SOFTMAX.");
        }

        Matrix error = output - expected;

        batchResult.errors.push_front(error);
        auto weightIt = weights_.rbegin();

        for (int i = transforms_.size() - 1; i >= 0; --i)
        {
            Transform transform = transforms_[i];
            Matrix input = io[i];
            Matrix output = io[i+1];

            switch (transform)
            {
            case DENSE:
            {
                Matrix weight = *weightIt;
                ++weightIt;

                if(weightIt == weights_.rend())
                {
                    break;
                }

                error = weight.transpose() * error;
            }
            break;
            case RELU:
               
                error = error.apply([&](int row, int col, int index, double value){
                    if(input.get(row, col) < 0)
                    {
                        return 0.0;
                    }

                    return value;
                });
 
                break;
            case SOFTMAX:
                
                break;
            }
        }

        batchResult.errors.push_front(error);

        return batchResult;
    }

     void NeuralNet::adjust(BatchResult batchResult)
     {
         std::lock_guard guard(mtxWeights_);

         std::cout << weights_.size() << std::endl;
         std::cout << biases_.size() << std::endl;
         std::cout << batchResult.errors.size() << std::endl;
         std::cout << batchResult.io.size() << std::endl;

         for(int i = 0; i < weights_.size(); ++i)
         {
             Matrix weight = weights_[i];
             Matrix bias = biases_[i];
             Matrix error = batchResult.errors[i];
             Matrix input = batchResult.io[i];

             std::cout << "\n\n" << i << std::endl;
             std::cout << "WEIGHT: " << weight.rows() << "x" << weight.cols() << std::endl;
             std::cout << "BIAS: " << bias.rows() << "x" << bias.cols() << std::endl;
             std::cout << "ERROR: " << error.rows() << "x" << error.cols() << std::endl;
             std::cout << "INPUT: " << input.rows() << "x" << input.cols() << std::endl;
         }
     }

    NeuralNet::NeuralNet(std::vector<int> layerSizes)
    {
        if (layerSizes.size() < 2)
        {
            throw std::invalid_argument("Must be at least 2 layers.");
        }

        for (int i = 1; i < layerSizes.size(); ++i)
        {
            int rows = layerSizes[i];
            int cols = layerSizes[i - 1];

            add(DENSE, rows, cols);
            add(RELU);
        }

        transforms_[transforms_.size() - 1] = SOFTMAX;
    }
}