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

int main()
{
    int inputSize = 784;
    int outputSize = 10;
    int batchSize = 32;

    std::string inputDir = "../data";

/*
    MNISTLoader trainLoader(batchSize, inputDir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    MNISTLoader evalLoader(batchSize, inputDir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
*/

    ImageWriter imageWriter("../data");
    imageWriter.write("../images");
    return 0;


    /*
    TestLoader trainingLoader(60000, inputSize, outputSize, batchSize);
    TestLoader evalLoader(10000, inputSize, outputSize, batchSize);

    TrainingData trainingData = trainingLoader.load();
    TrainingData evalData = evalLoader.load();
    

    NeuralNet neuralNet;
    neuralNet.add(NeuralNet::DENSE, 200, inputSize);
    neuralNet.add(NeuralNet::RELU);
    neuralNet.add(NeuralNet::DENSE, outputSize);
    neuralNet.add(NeuralNet::SOFTMAX);
    neuralNet.setScaleInitialWeights(0.2);
    neuralNet.setEpochs(20);
    neuralNet.setLearningRates(0.02, 0.001);

    neuralNet.fit(trainingData.input, trainingData.expected);
    double accuracy = neuralNet.evaluate(evalData.input, evalData.expected);

    cout << std::fixed << std::setprecision(2) << "Accuracy: " << accuracy << "%" << std::endl;

    cout << neuralNet << endl;
*/

    return 0;
}