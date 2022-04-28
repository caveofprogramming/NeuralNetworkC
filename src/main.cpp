#include <iostream>
#include <random>
#include <vector>
#include "matrix.h"
#include "matrixfunctions.h"
#include "neuralnet.h"

using namespace std;
using namespace cave;

#include "testloader.h"

struct Test
{
    Test(double value, std::vector<double> args)
    {
        check = value;

        for (auto v : args)
        {
            values.push_back(v);
        }
    }

    friend std::ostream& operator<<(std::ostream &out, Test &t)
    {
        out << "TEST: " << t.check << ": ";

        for(auto &v: t.values)
        {
            out << v << " ";
        }
        out << std::endl;

        return out;
    }

    double check;
    std::vector<double> values;
};

int main()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    int batchSize = 5;
    int items = 4;
    int inputSize = 5;
    int outputSize = 6;

    TestLoader testLoader(items, inputSize, outputSize, batchSize);

    NeuralNet neuralNet;
    neuralNet.add(NeuralNet::DENSE, 3, inputSize);
    neuralNet.add(NeuralNet::RELU);
    neuralNet.add(NeuralNet::DENSE, outputSize);
    neuralNet.add(NeuralNet::SOFTMAX);
    cout << neuralNet << endl;

    auto data = generateTestData(items, inputSize, outputSize);

    Matrix &input = data.input;
    Matrix &expected = data.output;

    Matrix output = neuralNet.predict(input);

    // clang-format off
    Matrix grad = gradient(&input, [&]()
    {
        Matrix output = neuralNet.predict(input);

        return crossEntropy(output, expected); 
    });
    // clang-format on

    neuralNet.fit(input, expected);

    std::cout << "\n\nApproximated  grad:\n"
              << grad << std::endl;

    return 0;
}