#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "matrix.h"
#include "matrixfunctions.h"
#include "neuralnet.h"
#include "blockingqueue.h"
#include "threadpool.h"

using namespace std;
using namespace cave;

#include "testloader.h"

mutex g_mtx;
int threadCount = 0;

int work(int id)
{
    int seconds = 3; //int((5.0 * rand()) / RAND_MAX + 3);
    this_thread::sleep_for(chrono::seconds(seconds));

    return id;
}

int main()
{
    cave::ThreadPool<int> threadpool(3);

    for (int i = 0; i < 10; i++)
    {
        threadpool.submit(std::bind(work, i));
    }

    threadpool.start();

    for (int i = 0; i < threadpool.size(); i++)
    {
        int result = threadpool.get();
        cout << "Result: " << result << endl;
    }
    /*
    BlockingQueue<shared_future<int> > futures(1);

    thread t([&]()
             {
        for (int i = 0; i < 20; i++)
        {
            shared_future<int> f = async(launch::async, work, i);
            futures.push(f);
        } });

    for (int i = 0; i < 20; i++)
    {
        shared_future<int> f = futures.front();
        int value = f.get();

        unique_lock<mutex> lock(g_mtx);
        --threadCount;
        lock.unlock();

        futures.pop();
        cout << "Returned: " << value << endl;
    }

    t.join();
    */

    return 0;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    int batchSize = 32;
    int items = 4;
    int inputSize = 5;
    int outputSize = 6;

    TestLoader testLoader(60000, inputSize, outputSize, batchSize);
    TestLoader evalLoader(10000, inputSize, outputSize, batchSize);

    NeuralNet neuralNet;
    neuralNet.add(NeuralNet::DENSE, 3, inputSize);
    neuralNet.add(NeuralNet::RELU);
    neuralNet.add(NeuralNet::DENSE, outputSize);
    neuralNet.add(NeuralNet::SOFTMAX);
    cout << neuralNet << endl;

    neuralNet.fit(testLoader, evalLoader);

    return 0;
}