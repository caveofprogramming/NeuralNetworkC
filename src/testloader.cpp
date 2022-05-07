#include "testloader.h"
#include "matrixfunctions.h"
#include <iostream>

#include <math.h>

namespace cave
{
    TestLoader::TestLoader(int items, int inputSize, int outputSize, int batchSize)
    {
        metaData_.items = items;
        metaData_.inputSize = inputSize;
        metaData_.outputSize = outputSize;
        metaData_.batchSize = batchSize;
        metaData_.numberBatches = ceil(double(items) / batchSize);
    }

    MetaData &TestLoader::open()
    {
        return metaData_;
    }

    MetaData &TestLoader::getMetaData()
    {
        return metaData_;
    }

    BatchData TestLoader::getBatch()
    {
        std::lock_guard<std::mutex> guard(mtxRead_);

        int itemsRead = std::min(metaData_.items - totalItemsRead_, metaData_.batchSize);
        
        auto testData = generateTestData(itemsRead, metaData_.inputSize, metaData_.outputSize);
       
        Matrix input = testData.input.transpose();
        Matrix output = testData.output.transpose();

        BatchData batchData;

        batchData.input = input.get();
        batchData.expected = output.get();

        totalItemsRead_ += itemsRead;

        batchData.numberRead = itemsRead;
        
        return batchData;
    }

    void TestLoader::close()
    {
        totalItemsRead_ = 0;
    }
}
