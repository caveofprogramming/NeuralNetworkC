#include <iostream>
#include <random>
#include "matrix.h"

using namespace std;
using namespace cave;

int main() {

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    
    Matrix m1(5, 5, [&](){
        return distribution(generator);
    });

    cout << m1 << endl;

    return 0;
}