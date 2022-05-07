#pragma once

#include <vector>

namespace cave
{
    struct MetaData
    {
        int items{0};
        int inputSize{0};
        int outputSize{0};
        int batchSize{0};
        int numberBatches{0};
    };

    struct BatchData
    {
        int numberRead;
        std::vector<double> input;
        std::vector<double> expected;
    };

    class Loader
    {
    public:
        virtual MetaData &open() = 0;
        virtual MetaData &getMetaData() = 0;
        virtual BatchData getBatch() = 0;
        virtual void close() = 0;
    };
}