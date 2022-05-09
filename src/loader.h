#pragma once

#include <vector>

#include "matrix.h"

namespace cave
{
    struct TrainingData
    {
        std::vector<Matrix> input;
        std::vector<Matrix> expected;
    };

    class Loader
    {
    public:
        virtual TrainingData load() = 0;
    };
}