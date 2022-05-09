#pragma once

#include <mutex>
#include "loader.h"

namespace cave
{
    class TestLoader : public Loader
    {
    private:
        int items_;
        int inputSize_;
        int outputSize_;
        int batchSize_;

    public:
        TestLoader(int items, int inputSize, int outputSize, int batchSize)
            : items_(items), inputSize_(inputSize), outputSize_{outputSize}, batchSize_{batchSize}
        {
        }
        
        TrainingData load();
    };
}
