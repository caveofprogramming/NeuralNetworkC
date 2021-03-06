#include "matrix.h"
#include <utility>
#include <functional>
#include <string>

namespace cave
{
    struct IO
    {
        IO(Matrix input, Matrix output)
        {
            this->input = input;
            this->output = output;
        }

        Matrix input;
        Matrix output;
    };

    Matrix relu(Matrix &input);
    Matrix softmax(Matrix &input);
    IO generateTestData(int items, int inputSize, int outputSize);
    Matrix crossEntropy(Matrix &actual, Matrix &expected);
    Matrix square(Matrix input);
    Matrix getGreatestRowNumbers(Matrix &input);
    Matrix gradient(Matrix *input, std::function<Matrix()> func);
    Matrix incrementElement(const Matrix &m, int row, int col, double value);
    double numberCorrect(const Matrix &actual, Matrix &expected);
    std::vector<bool> getCorrect(const Matrix &actual, Matrix &expected);
    std::string toString(Matrix &m);
}