#include <cmath>

#include "neuralnettest.h"
#include "matrixfunctions.h"

namespace cave
{

    TestLoader NeuralNetTest::getTestLoader()
    {
        return TestLoader(60000, inputSize_, outputSize_, batchSize_);
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
        neuralNet_.setLearningRates(0.02, 0.001);
    }

    bool NeuralNetTest::all()
    {

        std::cout << "Testing backprop ... " << std::flush;
        bool backpropPassed = testBackprop();
        std::cout << (backpropPassed ? "passed" : "failed") << std::endl;

        std::cout << "Testing adjust ... " << std::flush;
        bool adjustPassed = testAdjust();
        std::cout << (adjustPassed ? "passed" : "failed") << std::endl;

        bool passed = backpropPassed && adjustPassed;

        if(passed)
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
        return true;
    }

    bool NeuralNetTest::testBackprop()
    {
        TestLoader loader = getTestLoader();

        MetaData metaData = loader.open();

        BatchData batchData = loader.getBatch();
        Matrix input(metaData.inputSize, batchData.batchItemsRead, batchData.input, false);
        Matrix expected(metaData.outputSize, batchData.batchItemsRead, batchData.expected, false);
        Matrix inputCopy = input.clone();

        BatchResult result = neuralNet_.runForwards(inputCopy);
        neuralNet_.runBackwards(result, expected, true);

        Matrix &inputError = result.errors.front();

        Matrix approximatedError = gradient(&input, [&]()
                                            {

            Matrix inputCopy = input.clone();
            BatchResult batchResult = neuralNet_.runForwards(inputCopy);

            return crossEntropy(batchResult.io.back(), expected); });

        if (inputError != approximatedError)
        {
            std::cerr << "Input error: calculated and approximated don't match." << std::endl;

            std::cerr << "\nCalculated:\n"
                      << inputError << std::endl;
            std::cerr << "\nApproximated:\n"
                      << approximatedError << std::endl;
            return false;
        }

        loader.close();

        if (std::abs(inputError.sum()) < 0.0001)
        {
            std::cout << "Input error contained only zero. Consider re-running test." << std::endl;
        }

        return true;
    }
}