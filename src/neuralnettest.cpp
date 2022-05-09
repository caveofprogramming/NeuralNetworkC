#include <cmath>

#include "neuralnettest.h"
#include "matrixfunctions.h"

namespace cave
{

    TestLoader NeuralNetTest::getTestLoader(int items)
    {
        return TestLoader(items, inputSize_, outputSize_, batchSize_);
    }

    void NeuralNetTest::configureNeuralNet()
    {
        neuralNet_.add(NeuralNet::DENSE, 100, inputSize_);
        neuralNet_.add(NeuralNet::RELU);
        neuralNet_.add(NeuralNet::DENSE, 50);
        neuralNet_.add(NeuralNet::RELU);
        neuralNet_.add(NeuralNet::DENSE, outputSize_);
        neuralNet_.add(NeuralNet::SOFTMAX);
        neuralNet_.setEpochs(20);
        neuralNet_.setThreads(4);
        neuralNet_.setLearningRates(0.02, 0.001);
    }

    bool NeuralNetTest::all()
    {

        std::cout << "Testing backprop ... " << std::flush;
        bool backpropPassed = testBackprop();
        std::cout << (backpropPassed ? "passed" : "failed") << std::endl;

        std::cout << "Testing adjust ... " << std::endl;
        neuralNet_.setEpochs(1);
        bool adjustPassed = testAdjust();
        std::cout << "\n"
                  << (adjustPassed ? "passed" : "failed") << std::endl;

        bool passed = backpropPassed && adjustPassed;

        if (passed)
        {
            std::cout << "All tests passed." << std::endl;
        }
        else
        {
            std::cerr << "Some tests failed." << std::endl;
        }

        return passed;
    }

    bool NeuralNetTest::testAdjust()
    {
        TestLoader trainingLoader = getTestLoader(60000);
        TestLoader evalLoader = getTestLoader(10000);

        TrainingData trainingData = trainingLoader.load();
        TrainingData evalData = evalLoader.load();

        neuralNet_.fit(trainingData.input, trainingData.expected);

        Matrix result = neuralNet_.predict(evalData.input[0]);

        int correct = numberCorrect(result, evalData.expected[0]);

        if (double(correct) / result.cols() > 0.96)
        {
            return true;
        }

        return false;
    }

    bool NeuralNetTest::testBackprop()
    {
        TestLoader loader = getTestLoader(1000);

        TrainingData data = loader.load();

        Matrix &input = data.input[0];
        Matrix &expected = data.expected[0];
        BatchResult result;

        neuralNet_.runForwards(result, input);
        neuralNet_.runBackwards(result, data.expected[0]);

        Matrix &inputError = result.errors.front();

        // clang-format off
        Matrix approximatedError = gradient(&input, [&]()
        {
            BatchResult result;
            
            neuralNet_.runForwards(result, input);

            return crossEntropy(result.io.back(), expected); 
        });
        // clang-format on

        if (inputError != approximatedError)
        {
            std::cerr << "Input error: calculated and approximated don't match." << std::endl;

            std::cerr << "\nCalculated:\n"
                      << inputError << std::endl;
            std::cerr << "\nApproximated:\n"
                      << approximatedError << std::endl;
            return false;
        }

        if (std::abs(inputError.sum()) < 0.0001)
        {
            std::cout << "Input error contained only zero. Consider re-running test." << std::endl;
        }

        return true;
    }
}