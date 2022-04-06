#pragma once

#include "loader.h"

namespace cave
{
    class TestLoader : public Loader
    {
    private:
        MetaData metaData_;
        int totalItemsRead_{0};

    public:
        TestLoader(int items, int inputSize, int outputSize, int batchSize);

        MetaData &open();
        MetaData &getMetaData();
        BatchData getBatch();
        void close();
    };
}
