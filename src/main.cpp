#include <iostream>
#include <random>
#include "matrix.h"
#include "matrixfunctions.h"
#include "neuralnet.h"

using namespace std;
using namespace cave;

#include "testloader.h"

int main() {

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    
    int batchSize = 5;
    int items = 12;
    int inputSize = 3;
    int outputSize = 5;

    TestLoader testLoader(items, inputSize, outputSize, batchSize);

    NeuralNet neuralNet({inputSize, 3, outputSize});

    cout << neuralNet << endl;
    

    return 0;
}