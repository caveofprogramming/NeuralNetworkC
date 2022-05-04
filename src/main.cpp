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

using namespace std;
using namespace cave;

#include "testloader.h"

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
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    int inputSize = 10; // 768;
    int outputSize = 3;
    int batchSize = 32;

    TestLoader testLoader(60000, inputSize, outputSize, batchSize);
    TestLoader evalLoader(10000, inputSize, outputSize, batchSize);

    NeuralNet neuralNet;
    neuralNet.add(NeuralNet::DENSE, 100, inputSize);
    neuralNet.add(NeuralNet::RELU);
    neuralNet.add(NeuralNet::DENSE, 50);
    neuralNet.add(NeuralNet::RELU);
    neuralNet.add(NeuralNet::DENSE, outputSize);
    neuralNet.add(NeuralNet::SOFTMAX);
    neuralNet.setEpochs(20);
    neuralNet.setLearningRates(0.02, 0.001);

    cout << neuralNet << endl;

    NeuralNetTest test;
    test.all();
    // neuralNet.fit(testLoader, evalLoader);

    return 0;
}