#include "matrixfunctions.h"

#include <cmath>
#include <utility>
#include <random>
#include <iostream>

namespace cave
{
    double numberCorrect(const Matrix &actual, Matrix &expected)
    {
        Matrix actualLargest = actual.largestRowIndexes();
        Matrix expectedLargest = expected.largestRowIndexes();

        int correct = 0;

        for(int i = 0; i < actualLargest.cols(); ++i)
        {
            if(std::abs(actualLargest.get(i) - expectedLargest.get(i)) < 0.1)
            {
                ++correct;
            }
        }

        return correct;
    }

    Matrix gradient(Matrix *input, std::function<Matrix()> func)
    {
        const double inc = 0.000001;

        Matrix result(input->rows(), input->cols());

        Matrix output1 = func();

        for (int row = 0; row < input->rows(); ++row)
        {
            for (int col = 0; col < input->cols(); ++col)
            {
                double value = input->get(row, col);

                input->set(row, col, value + inc);

                Matrix output2 = func();

                double rate = (output2.get(col) - output1.get(col)) / inc;
                result.set(row, col, rate);

                if (abs(rate) > 2E5)
                {
                    std::cout << "output1: " << output1.get(row, col) << std::endl;
                    std::cout << "output2: " << output2.get(row, col) << std::endl;
                    std::cout << "output::: " << output1 << std::endl;
                    std::cout << "diff: " << output2.get(row, col) - output1.get(row, col) << std::endl;
                    std::cout << "inc: " << inc << std::endl;
                    std::cout << "rate: " << rate << std::endl;
                    std::cout << "\n"
                              << std::endl;
                }

                input->set(row, col, value);
            }
        }

        return result;
    }

    Matrix getGreatestRowNumbers(Matrix &input)
    {
        Matrix result(1, input.cols());

        std::vector<double> maxValues(input.cols());

        input.forEach([&](int row, int col, int index, double value)
                      {
            if(value > maxValues[col])
            {
                maxValues[col] = value;
                result[col] = row;
            } });

        return result;
    }

    Matrix square(Matrix input)
    {
        Matrix result(input.rows(), input.cols());

        input.forEach([&](int row, int col, int index, double value)
                      { result.set(index, value * value); });
        return result;
    }

    Matrix crossEntropy(Matrix &actual, Matrix &expected)
    {
        Matrix result(1, actual.cols());

        Matrix biggestRows = getGreatestRowNumbers(expected);

        for (int col = 0; col < actual.cols(); ++col)
        {
            int activeRow = biggestRows[col];
            double actualValue = actual.get(activeRow, col);
            result[col] = -std::log(actualValue);
        }

        return result;
    }

    IO generateTestData(int items, int inputSize, int outputSize)
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

        return IO(std::move(input), std::move(output));
    }

    Matrix relu(Matrix &input)
    {
        return Matrix(input.rows(), input.cols(), [&](int index)
                      { return input[index] > 0 ? input[index] : 0; });
    }

    Matrix softmax(Matrix &input)
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