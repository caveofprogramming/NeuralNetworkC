#pragma once

#include <future>
#include <functional>
#include <algorithm>
#include <atomic>
#include <sstream>
#include <queue>

#include "threadpool.h"

namespace cave
{
    template <typename E>
    class ThreadPool
    {
    private:
        int threads_{0};
        int submissions_{0};
        std::queue<E> results_;
        std::queue<std::function<E()> > work_;


        std::mutex mtxWork_;
        std::mutex mtxResults_;
        std::condition_variable cond_;
        std::vector<std::shared_future<void>> futures_;

    private:
        void produce()
        {
            while (true)
            {
                std::unique_lock<std::mutex> workLock(mtxWork_);

                if (work_.size() == 0)
                {
                    break;
                }

                auto func = work_.front();
                work_.pop();

                workLock.unlock();

                E result = func();

                std::unique_lock<std::mutex> resultsLock(mtxResults_);
                results_.push(result);
                resultsLock.unlock();
                cond_.notify_one();
            }
        }

    public:
        ThreadPool(int threads) : threads_(threads)
        {
        }

        int size()
        {
            return submissions_;
        }

        void start()
        {
            for (int i = 0; i < threads_; ++i)
            {
                std::shared_future<void> future = std::async(std::launch::async, &ThreadPool::produce, this);
                futures_.push_back(future);
            }
        }

        void submit(std::function<E()> func)
        {
            ++submissions_;
            work_.push(func);
        }

        E get()
        {
            std::unique_lock<std::mutex> lock(mtxResults_);

            cond_.wait(lock, [this]()
                       { return results_.size() > 0; });

            E result = results_.front();
            results_.pop();
            return result;
        }
    };
}