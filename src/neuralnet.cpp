#include "neuralnet.h"

#include <random>
#include <iostream>
#include <exception>
#include <mutex>
#include <cmath>
#include <iomanip>

#include "blockingqueue.h"
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
                Matrix &weight = neuralNet.weights_[weightIndex++];
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
            weightIndices_.push_back(transforms_.size());

            std::default_random_engine generator;
            std::random_device rd;
            generator.seed(rd());
            std::normal_distribution<double> normal(0, 1);

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

            weights_.push_back(std::move(weight));
            biases_.push_back(std::move(bias));
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
        Matrix in = input.clone();

        if (weights_.size() > 0)
        {
            assert(input.rows() == weights_[0].cols());
        }

        BatchResult result = runForwards(in);

        return std::move(result.io.back());
    }

    std::vector<double> NeuralNet::predict(std::vector<double> input)
    {
        if (weights_.size() > 0)
        {
            assert(input.size() == weights_[0].cols());
        }

        Matrix mInput(input.size(), 1, input, false);
        BatchResult result = runForwards(mInput);

        return result.io.back().get();
    }

    void NeuralNet::runEpoch(Loader &loader, bool trainingMode)
    {
        MetaData metaData = loader.open();

        // std::cout << metaData.numberBatches <<  std::endl;

        double totalLoss = 0;
        int correct = 0;
        int totalItems = 0;

        int printDot = std::ceil(double(metaData.numberBatches) / 30);

        for (int i = 0; i < metaData.numberBatches; ++i)
        {
            BatchData batchData = loader.getBatch();
            Matrix input(metaData.inputSize, batchData.batchItemsRead, batchData.input, false);
            Matrix expected(metaData.outputSize, batchData.batchItemsRead, batchData.expected, false);

            totalItems += input.cols();

            BatchResult result = runForwards(input);

            if (trainingMode)
            {
                // std::cout << "\nLoss1: " << crossEntropy(result.io.back(), expected).rowMeans().get(0) << std::endl;
                runBackwards(result, expected);
                adjust(result, learningRate_);

                if (i % printDot == 0)
                {
                    std::cout << "." << std::flush;
                }
            }
            else
            {
                // std::cout << "\nLoss2: " << crossEntropy(result.io.back(), expected).rowMeans().get(0) << std::endl;

                correct += numberCorrect(result.io.back(), expected);
                // std::cout << "row mean: " << crossEntropy(result.io.back(), expected).rowMeans() << std::endl;
                totalLoss += crossEntropy(result.io.back(), expected).rowMeans().get(0);
            }
        }

        if (!trainingMode)
        {
            double averageLoss = totalLoss / metaData.numberBatches;

            std::cout << " Loss: " << averageLoss << " -- percent correct: " << (100.0 * correct) / totalItems << std::endl;
        }

        loader.close();
    }

    void NeuralNet::fit(Loader &trainingLoader, Loader &evaluationLoader)
    {
        learningRate_ = initialLearningRate_;

        MetaData metaData = trainingLoader.open();

        if (weights_.size() > 0)
        {
            assert(metaData.inputSize == weights_[0].cols());
        }

        for (int epoch = 0; epoch < epochs_; ++epoch)
        {
            std::cout << "Epoch " << std::setw(3) << std::fixed << std::setprecision(2) << (epoch + 1) << " " << std::flush;

            runEpoch(trainingLoader, true);
            runEpoch(evaluationLoader, false);
            learningRate_ -= (initialLearningRate_ - finalLearningRate_) / epochs_;
        }
    }

    BatchResult NeuralNet::runForwards(Matrix &input)
    {
        int weightIndex = 0;

        BatchResult result;

        result.io.push_back(std::move(input));

        int ioIndex = 0;

        for (Transform transform : transforms_)
        {
            Matrix &output = result.io.back();

            switch (transform)
            {
            case DENSE:
            {
                Matrix &weight = weights_[weightIndex];
                Matrix &bias = biases_[weightIndex];

                input = (weight * result.io.back());

                input.modify([&](int row, int col, int index, double value)
                             { return value + bias.get(row); });

                ++weightIndex;
            }
            break;
            case RELU:
                input = relu(result.io.back());
                break;
            case SOFTMAX:
                input = softmax(result.io.back());
                break;
            }

            result.io.push_back(std::move(input));

            ++ioIndex;
        }

        return result;
    }

    void NeuralNet::runBackwards(BatchResult &batchResult, Matrix &expected, bool bInputError)
    {
        auto &io = batchResult.io;

        Matrix &output = io.back();

        if (transforms_.back() != SOFTMAX)
        {
            throw std::logic_error("Final transform must be SOFTMAX.");
        }

        auto weightIt = weights_.rbegin();

        Matrix error;

        for (int i = transforms_.size() - 1; i >= 0; --i)
        {
            Transform transform = transforms_[i];
            Matrix &input = io[i];
            Matrix &output = io[i + 1];

            switch (transform)
            {
            case DENSE:
            {
                Matrix &weight = *weightIt;
                ++weightIt;

                if (bInputError || weightIt != weights_.rend())
                {
                    error = weight.transpose() * batchResult.errors.front();
                }
                else
                {
                    error = Matrix();
                }
            }
            break;
            case RELU:

                // clang-format off
                error = batchResult.errors.front().apply([&](int row, int col, int index, double value)
                {
                    if(input.get(row, col) < 0)
                    {
                        return 0.0;
                    }

                    return value; 
                });
                // clang-format on

                break;
            case SOFTMAX:
                error = output - expected;
                break;
            }

            batchResult.errors.push_front(std::move(error));
        }
    }

    void NeuralNet::adjust(BatchResult &batchResult, double learningRate)
    {
        std::lock_guard<std::mutex> lock(mtxWeights_);

        for (int i = 0; i < weights_.size(); ++i)
        {
            int weightIndex = weightIndices_[i];

            Matrix &weight = weights_[i];
            Matrix &bias = biases_[i];
            Matrix &error = batchResult.errors[weightIndex + 1];
            Matrix &input = batchResult.io[weightIndex];

            Matrix biasAdjust = learningRate * error.rowMeans();
            Matrix weightAdjust = (double(learningRate) / input.cols()) * (error * input.transpose());

            bias.modify([&](int row, int col, int index, double value)
                        { return value - biasAdjust.get(row); });

            //std::cout << std::setprecision(6) << "weight sum1: " << weight.sum() << std::endl;

            weight.modify([&](int row, int col, int index, double value)
                          { return value - weightAdjust.get(index); });

            //std::cout << std::setprecision(6) << "weight sum2: " << weight.sum() << std::endl;
        }

        // std::cout << "error\n"
        //           << batchResult.errors.front() << std::endl;
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