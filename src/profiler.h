#pragma once

#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <ostream>
#include <stdexcept>
#include <mutex>

namespace cave
{
    struct Timing
    {
        std::string function;
        std::clock_t start;
        std::clock_t end;
    };

    class Profiler
    {
    private:
        bool active_{false};
        std::vector<Timing> timings_;
        std::map<std::string, double> report_;
        std::mutex mtx_;

    public:
        void activate() { active_ = true; };

        Timing start(std::string name);
        void end(Timing &timing);
        friend std::ostream &operator<<(std::ostream &out, Profiler &prof);
    };
}

extern cave::Profiler gProfiler;