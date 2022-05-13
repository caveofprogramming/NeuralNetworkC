#include "neuralnet.h"

#include <random>
#include <iostream>
#include <exception>
#include <assert.h>
#include <mutex>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>

#include "blockingqueue.h"
#include "matrixfunctions.h"
#include "threadpool.h"
#include "profiler.h"
#include "fileutil.h"

namespace cave
{
    std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet)
    {
        out << "Threads: " << neuralNet.threads_ << std::endl;
        out << "Epochs: " << neuralNet.epochs_ << std::endl;
        out << "Initial learning rate: " << neuralNet.initialLearningRate_ << std::endl;
        out << "Final learning rate: " << neuralNet.finalLearningRate_ << std::endl;
        out << "Weight scale: " << neuralNet.scaleInitialWeights_ << std::endl;

        if (neuralNet.weights_.size() > 0)
        {
            out << "\nLayers:\n\n";
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

    void NeuralNet::save(std::string file)
    {
        std::ofstream out;

        out.open(file, std::ios::binary);


        if (!out.is_open())
        {
            throw FileException("Unable to open file");
        }

        cave::saveValueVector<Transform>(out, transforms_);

        cave::saveSerializableVector<Matrix>(out, weights_);
        cave::saveSerializableVector<Matrix>(out, biases_);
        cave::saveValueVector<int>(out, weightIndices_);

        cave::saveValue<double>(out, scaleInitialWeights_);
        cave::saveValue<double>(out, initialLearningRate_);
        cave::saveValue<double>(out, finalLearningRate_);
        cave::saveValue<int>(out, epochs_);
        cave::saveValue<int>(out, threads_);

        out.close();

        if (!out)
        {
           throw FileException("Unable to close file");
        }

    }

    void NeuralNet::load(std::string file)
    {
        std::ifstream in;

        in.open(file, std::ios::binary);

        if (!in.is_open())
        {
            throw FileException("Unable to open file");
        }
 
        transforms_ = cave::loadValueVector<Transform>(in);
        weights_ = cave::loadSerializableVector<Matrix>(in);
        biases_ = cave::loadSerializableVector<Matrix>(in);
        weightIndices_ = cave::loadValueVector<int>(in);
        scaleInitialWeights_ = cave::loadValue<double>(in);
        initialLearningRate_ = cave::loadValue<double>(in);
        finalLearningRate_ = cave::loadValue<double>(in);
        epochs_ = cave::loadValue<int>(in);
        threads_ = cave::loadValue<int>(in);
        in.close();

        if (!in)
        {
            throw FileException("Unable to close file");
        }
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

            weights_.push_back(weight);
            biases_.push_back(bias);
        }

        transforms_.push_back(transform);
    }

    Matrix NeuralNet::loss(BatchResult &result, Matrix &expecteds)
    {
        Matrix &output = result.io.back();

        auto losses = crossEntropy(output, expecteds);

        return losses;
    }

    Matrix NeuralNet::predict(Matrix &input)
    {
        Matrix in = input.clone();

        if (weights_.size() > 0)
        {
            assert(input.rows() == weights_[0].cols());
        }

        BatchResult result;
        runForwards(result, in);

        return result.io.back();
    }

    BatchResult NeuralNet::runBatch(Matrix &input, Matrix &expected)
    {
        BatchResult batchResult;

        batchResult.numberItems = input.cols();

        runForwards(batchResult, input);

        runBackwards(batchResult, expected);
        adjust(batchResult, learningRate_);

        batchResult.numberCorrect = numberCorrect(batchResult.io.back(), expected);
        batchResult.totalLoss = crossEntropy(batchResult.io.back(), expected).rowSums().get(0);

        return batchResult;
    }

    void NeuralNet::runEpoch(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds)
    {
        double totalLoss = 0;
        int totalCorrect = 0;
        int totalItems = 0;

        int printDot = std::ceil(inputs.size() / 30);

        cave::ThreadPool<BatchResult> threadPool(threads_);

        for (int i = 0; i < inputs.size(); ++i)
        {
            // clang-format off
            threadPool.submit([this, i, &inputs, &expecteds]()
            { 
                return runBatch(inputs[i], expecteds[i]); 
            });
            // clang-format on
        }

        threadPool.start();

        for (int i = 0; i < inputs.size(); ++i)
        {
            BatchResult result = threadPool.get();

            if (i % printDot == 0)
            {
                std::cout << "." << std::flush;
            }

            totalItems += result.numberItems;
            totalCorrect += result.numberCorrect;
            totalLoss += result.totalLoss;
        }

        double averageLoss = totalLoss / totalItems;

        std::cout << " Loss: " << averageLoss << " -- percent correct: "
                  << ((100.0 * totalCorrect) / totalItems) << "%: ";
    }

    double NeuralNet::evaluate(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds)
    {
        int totalCorrect = 0;
        int totalItems = 0;

        for (int i = 0; i < inputs.size(); ++i)
        {
            BatchResult result = runBatch(inputs[i], expecteds[i]);

            totalCorrect += result.numberCorrect;
            totalItems += result.numberItems;
        }

        return double(totalCorrect) / totalItems;
    }

    void NeuralNet::fit(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds)
    {
        auto timing = gProfiler.start("fit");

        learningRate_ = initialLearningRate_;

        if (weights_.size() > 0)
        {
            int inputSize = inputs[0].rows();

            if (inputSize != weights_[0].cols())
            {
                std::stringstream ss;

                ss << "Input size is " << inputSize;
                ss << " but first dense layer has " << weights_[0].cols();
                ss << " columns; mismatch.";
                throw std::logic_error(ss.str());
            }
        }

        for (int epoch = 0; epoch < epochs_; ++epoch)
        {
            std::cout << "Epoch " << std::setw(3) << std::fixed << std::setprecision(2) << (epoch + 1) << " " << std::flush;

            auto start = std::chrono::high_resolution_clock::now();

            runEpoch(inputs, expecteds);

            auto finish = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);

            std::cout << std::setprecision(1)
                      << duration.count() / 1000.0 << "s" << std::endl;

            learningRate_ -= (initialLearningRate_ - finalLearningRate_) / epochs_;
        }

        gProfiler.end(timing);
    }

    void NeuralNet::runForwards(BatchResult &result, Matrix &input)
    {
        auto timing = gProfiler.start("runForwards");
        int weightIndex = 0;

        result.io.push_back(input);

        auto timing1 = gProfiler.start("acquiring lock");
        std::unique_lock<std::mutex> lock(mtxWeights_);
        gProfiler.end(timing1);
        auto weights(weights_);
        lock.unlock();

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

                auto timing3 = gProfiler.start("weight * output");
                Matrix multiplicationResult = weight * output;
                gProfiler.end(timing3);

                result.io.push_back(multiplicationResult);

                result.io.back().modify([&](int row, int col, int index, double value)
                                        { return value + bias.get(row); });

                ++weightIndex;
            }
            break;
            case RELU:
                result.io.push_back(relu(output));
                break;
            case SOFTMAX:
                result.io.push_back(softmax(output));
                break;
            }

            ++ioIndex;
        }

        gProfiler.end(timing);
    }

    void NeuralNet::runBackwards(BatchResult &batchResult, Matrix &expecteds, bool bInputError)
    {
        auto timing = gProfiler.start("runBackwards");
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
                    std::unique_lock<std::mutex> lock(mtxWeights_);
                    Matrix transpose = weight.transpose();
                    lock.unlock();
                    error = transpose * batchResult.errors.front();
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
                assert(output.rows() == expecteds.rows() && "expecteds data has different size to output.");

                error = output - expecteds;
                break;
            }

            batchResult.errors.push_front(error);
        }

        gProfiler.end(timing);
    }

    void NeuralNet::adjust(BatchResult &batchResult, double learningRate)
    {
        auto timing = gProfiler.start("adjust");

        for (std::size_t i = 0; i < weights_.size(); ++i)
        {
            int weightIndex = weightIndices_[i];

            Matrix &error = batchResult.errors[weightIndex + 1];
            Matrix &input = batchResult.io[weightIndex];

            std::unique_lock<std::mutex> lock(mtxWeights_);
            biases_[i] -= learningRate * error.rowMeans();
            weights_[i] -= (double(learningRate) / input.cols()) * (error * input.transpose());
            lock.unlock();
        }

        gProfiler.end(timing);
    }

    NeuralNet::NeuralNet(std::vector<int> layerSizes)
    {
        if (layerSizes.size() < 2)
        {
            throw std::invalid_argument("Must be at least 2 layers.");
        }

        for (std::size_t i = 1; i < layerSizes.size(); ++i)
        {
            int rows = layerSizes[i];
            int cols = layerSizes[i - 1];

            add(DENSE, rows, cols);
            add(RELU);
        }

        transforms_[transforms_.size() - 1] = SOFTMAX;
    }
}