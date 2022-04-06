#include "matrix.h"
#include <utility>

namespace cave
{
    Matrix relu(Matrix input);
    Matrix softmax(Matrix input);
    std::pair<Matrix, Matrix> generateTestData(int items, int inputSize, int outputSize);
    Matrix crossEntropy(Matrix actual, Matrix expected);
    Matrix getGreatestRowNumbers(Matrix input);
}