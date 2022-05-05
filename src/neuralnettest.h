#pragma once

#include "neuralnet.h"
#include "testloader.h"

namespace cave
{
    class NeuralNetTest
    {
    private:
        NeuralNet neuralNet_;

        int inputSize_{10};
        int outputSize_{3};
        int batchSize_{32};

        TestLoader getTestLoader(int items);
        void configureNeuralNet();
    public:
        NeuralNetTest()
        {
            configureNeuralNet();
        }

        bool testBackprop();
        bool testAdjust();
        bool all();
    };
}