#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <deque>
#include <mutex>
#include "matrix.h"

namespace cave
{
    class NeuralNetTest;

    struct BatchResult
    {
        BatchResult() {}

        BatchResult(BatchResult &&other)
        {
            io = std::move(other.io);
            errors = std::move(other.errors);

            numberItems = other.numberItems;
            numberCorrect = other.numberCorrect;
            totalLoss = other.totalLoss;
        }

        BatchResult &operator=(BatchResult &&other)
        {
            io = std::move(other.io);
            errors = std::move(other.errors);

            numberItems = other.numberItems;
            numberCorrect = other.numberCorrect;
            totalLoss = other.totalLoss;

            return *this;
        }

        std::vector<Matrix> io;
        std::deque<Matrix> errors;

        int numberItems{0};
        int numberCorrect{0};
        double totalLoss{0};
    };

    class NeuralNet
    {
    public:
        enum Transform
        {
            DENSE = 0,
            RELU = 1,
            SOFTMAX = 2,
        };
    private:
        std::mutex mtxWeights_;

        std::vector<std::string> transformNames_{"DENSE", "RELU", "SOFTMAX"};

        std::vector<Matrix> weights_;
        std::vector<Matrix> biases_;
        std::vector<int> weightIndices_;

        std::vector<Transform> transforms_;

        double scaleInitialWeights_{0.2};
        double initialLearningRate_{0.01};
        double finalLearningRate_{0.001};
        double learningRate_{0.01};
        bool errorAtInput_{false};

        int epochs_{20};
        int threads_{4};

    private: 
        void runForwards(BatchResult &batchResult, Matrix &input);
        void runBackwards(BatchResult &batchResult, Matrix &expecteds, bool bInputError = false);
        void adjust(BatchResult &batchResult, double learningRate);
        Matrix loss(BatchResult &result, Matrix &expecteds);
        void runEpoch(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds);
        BatchResult runBatch(Matrix &input, Matrix &expected);

    public:
        NeuralNet(){};
        NeuralNet(std::vector<int> layerSizes);

        void add(NeuralNet::Transform transform, int rows = 0, int cols = 0);
        void setScaleInitialWeights(double scale) { scaleInitialWeights_ = scale; };
        void setLearningRates(double initial, double final){ initialLearningRate_ = initial; finalLearningRate_ = final; };
        void fit(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds);
        double evaluate(std::vector<Matrix> &inputs, std::vector<Matrix> &expecteds);
        Matrix predict(Matrix &input);
        void setEpochs(int epochs) { epochs_ = epochs; }
        std::vector<double> predict(std::vector<double> input);
        Matrix &getWeight(int i) { return weights_[i]; };
        Matrix &getBias(int i) { return biases_[i]; };
        void setThreads(int threads){ threads_ = threads;}

        friend std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet);

        friend class cave::NeuralNetTest;
    };
}