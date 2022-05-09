#include "testloader.h"
#include "matrixfunctions.h"
#include <iostream>

#include <cmath>

namespace cave
{
    TrainingData TestLoader::load()
    {
        TrainingData trainingData;

        int totalItems = 0;

        int numberBatches = std::ceil(double(items_)/batchSize_);

        for(int batch = 0; batch < numberBatches; ++batch)
        {
            int itemsToRead = std::min(batchSize_, items_ - totalItems);

            auto testData = generateTestData(itemsToRead, inputSize_, outputSize_);

            trainingData.input.push_back(std::move(testData.input));
            trainingData.expected.push_back(std::move(testData.output));

            totalItems += itemsToRead;
        }

        return trainingData;
    }
}
