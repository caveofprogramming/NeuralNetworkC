#include "matrixfunctions.h"

#include <cmath>
#include <utility>
#include <random>
#include <iostream>

namespace cave
{

    Matrix getGreatestRowNumbers(Matrix input)
    {
        Matrix result(1, input.cols());

        std::vector<double> maxValues(input.cols());

        input.forEach([&](int row, int col, int index, double value){
            if(value > maxValues[col])
            {
                maxValues[col] = value;
                result[col] = row;
            }
        });

        return result;
    }

    Matrix crossEntropy(Matrix actual, Matrix expected)
    {
        Matrix result(1, actual.cols());

        Matrix biggestRows = getGreatestRowNumbers(expected);

        for(int col = 0; col < actual.cols(); ++col)
        {
            int activeRow = biggestRows[col];
            double actualValue = actual.get(activeRow, col);
            result[col] = -std::log(actualValue);
        }

        return result;
    }

    std::pair<Matrix, Matrix> generateTestData(int items, int inputSize, int outputSize)
    {
        std::default_random_engine generator;
        std::random_device rd;
        generator.seed(rd());

        std::uniform_int_distribution<int> uniform(1, outputSize);
        std::normal_distribution<double> normal(0, 1);

        Matrix input(inputSize, items);
        Matrix output(outputSize, items);
        for (int col = 0; col < items; col++)
        {
            int radius = uniform(generator);

            output.set(radius - 1, col, 1.0);

            double sumsquare = 0.0;

            for (int row = 0; row < inputSize; ++row)
            {
                double value = normal(generator);
                sumsquare += value * value;

                input.set(row, col, value);
            }

            double distance = std::sqrt(sumsquare);

            for (int row = 0; row < inputSize; ++row)
            {
                double value = input.get(row, col);
                input.set(row, col, radius * value / distance);
            }
        }
       
        return std::make_pair(input, output);
    }

    Matrix relu(Matrix input)
    {
        return Matrix(input.rows(), input.cols(), [&](int index)
                      { return input[index] > 0 ? input[index] : 0; });
    }

    Matrix softmax(Matrix input)
    {
        Matrix result(input.rows(), input.cols());

        input.forEach([&](int row, int col, int index, double value)
                      { result[index] = exp(value); });

        Matrix sum = result.sumColumns();

        result.modify([&](int row, int col, int index, double value)
                      { return value / sum[col]; });

        return result;
    }
}