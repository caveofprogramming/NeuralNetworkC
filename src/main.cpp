#include <iostream>
#include <random>
#include "matrix.h"
#include "matrixfunctions.h"

using namespace std;
using namespace cave;

int main() {

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    
    auto io = generateTestData(5, 4, 6);

    cout << io.first << endl;
    cout << io.second << endl;

    Matrix sumsquares(1, io.second.cols());

    io.first.forEach([&](int row, int col, int index, double value){
        sumsquares[col] += value * value;
    });

    cout << sumsquares << endl;

    return 0;
}