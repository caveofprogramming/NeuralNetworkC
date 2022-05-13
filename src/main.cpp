#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "matrix.h"
#include "matrixfunctions.h"
#include "neuralnet.h"
#include "blockingqueue.h"
#include "threadpool.h"
#include "neuralnettest.h"
#include "mnistloader.h"
#include "profiler.h"

using namespace std;
using namespace cave;

#include "testloader.h"
#include "imagewriter.h"

mutex g_mtx;
int threadCount = 0;

int work(int id)
{
    int seconds = 3; // int((5.0 * rand()) / RAND_MAX + 3);
    this_thread::sleep_for(chrono::seconds(seconds));

    return id;
}

int main(int argc, char *argv[])
{
    if (argc == 0)
    {
        std::cout << "Expected program name as first command line argument." << std::endl;
        return 0;
    }
    else if (argc > 3)
    {
        std::cout << "Too many arguments." << std::endl;
        return 0;
    }

    if (argc == 1)
    {
        // Got program name and nothing else.
        std::cout << "usage: " << argv[0] << " <data directory> [saved state file]" << std::endl;
        return 0;
    }

    std::string inputDir = argv[1];

    bool loadFromFile = false;
    std::string stateFile;

    if (argc == 3)
    {
        stateFile = argv[2];
        loadFromFile = true;
    }

    int inputSize = 784;
    int outputSize = 10;
    int batchSize = 32;

    MNISTLoader trainingLoader(batchSize, inputDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    MNISTLoader evalLoader(batchSize, inputDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    /*
    ImageWriter imageWriter("../data");
    imageWriter.write("../images");
    return 0;
    */

    /*
    NeuralNetTest test;
    test.all();
    return 0;
    */

    /*
    TestLoader trainingLoader(60000, inputSize, outputSize, batchSize);
    TestLoader evalLoader(10000, inputSize, outputSize, batchSize);
    */

    TrainingData trainingData = trainingLoader.load();
    TrainingData evalData = evalLoader.load();

    NeuralNet neuralNet;

    if (loadFromFile)
    {
        try
        {
            neuralNet.load(stateFile);
        }
        catch (const FileException &e)
        {
            std::cerr << "'" << stateFile << "': " << e.what() << std::endl;
            return 0;
        }
    }
    else
    {
        neuralNet.add(NeuralNet::DENSE, 200, inputSize);
        neuralNet.add(NeuralNet::RELU);
        neuralNet.add(NeuralNet::DENSE, outputSize);
        neuralNet.add(NeuralNet::SOFTMAX);
        neuralNet.setScaleInitialWeights(0.2);
        neuralNet.setEpochs(100);
        neuralNet.setThreads(4);
        neuralNet.setLearningRates(0.02, 0.001);
    }

    std::cout << "\n"
              << neuralNet << std::endl;

    neuralNet.fit(trainingData.input, trainingData.expected);

    double accuracy = neuralNet.evaluate(evalData.input, evalData.expected);

    cout << std::fixed << std::setprecision(2) << "\nAccuracy: " << 100.0 * accuracy << " %" << std::endl;

    cout << gProfiler << endl;


    std::string defaultFile = "default.ann";

    std::cout << "Saving to " << defaultFile << " ..." << std::flush;
    try
    {
        neuralNet.save(defaultFile);
    }
    catch (const FileException &e)
    {
        std::cout << "'" << defaultFile << "': " << e.what() << std::endl;
    }

    std::cout << " saved." << std::endl;

    return 0;
}