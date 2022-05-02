#pragma once

#include <mutex>
#include <queue>
#include <thread>

namespace cave
{
    template <typename E>
    class BlockingQueue
    {
    private:
        int size_;
        std::mutex mtx_;
        std::condition_variable cond_;
        std::queue<E> queue_;

    public:
        BlockingQueue(int size) : size_(size)
        {
        }

        void push(E e)
        {
            std::unique_lock<std::mutex> lock(mtx_);

            cond_.wait(lock, [this]()
                       { return queue_.size() < size_; });

            queue_.push(e);

            lock.unlock();
            cond_.notify_one();
        }

        E front()
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cond_.wait(lock, [this]()
                       { return !queue_.empty(); });

            return queue_.front();
        }

        void pop()
        {
            std::unique_lock<std::mutex> lock(mtx_);

            cond_.wait(lock, [this]()
                       { return !queue_.empty(); });

            queue_.pop();

            lock.unlock();
            cond_.notify_one();
        }

        int size()
        {
            std::lock_guard<std::mutex> lock(mtx_);
            return queue_.size();
        }
    };
}