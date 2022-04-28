#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <deque>
#include <mutex>
#include "matrix.h"
#include "loader.h"

namespace cave
{
    struct BatchResult
    {
        BatchResult() {}
        BatchResult(BatchResult &&other)
        {
            weightInputs = std::move(other.weightInputs);
            io = std::move(other.io);
            errors = std::move(other.errors);
        }

        BatchResult &operator=(BatchResult &&other)
        {
            weightInputs = std::move(other.weightInputs);
            io = std::move(other.io);
            errors = std::move(other.errors);

            return *this;
        }

        std::vector<int> weightInputs;
        std::vector<Matrix> io;
        std::deque<Matrix> errors;
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

        std::vector<Transform> transforms_;

        double scaleInitialWeights_ = 1.0;

    private: 
        BatchResult runForwards(Matrix &input);
        void runBackwards(BatchResult &batchResult, Matrix &expected);
        void adjust(BatchResult &batchResult);
        Matrix loss(BatchResult &result, Matrix &expected);

    public:
        NeuralNet(){};
        NeuralNet(std::vector<int> layerSizes);

        void add(NeuralNet::Transform transform, int rows = 0, int cols = 0);
        void setScaleInitialWeights(double scale) { scaleInitialWeights_ = scale; };
        void fit(Matrix &input, Matrix &expected);
        void fit(Loader &trainingLoader);
        Matrix predict(Matrix &input);
        std::vector<double> predict(std::vector<double> input);
        Matrix &getWeight(int i) { return weights_[i]; };
        Matrix &getBias(int i) { return biases_[i]; };

        friend std::ostream &operator<<(std::ostream &out, NeuralNet &neuralNet);
    };
}