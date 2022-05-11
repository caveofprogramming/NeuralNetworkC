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
        std::vector<Timing> timings_;
        std::map<std::string, double> report_;
        std::mutex mtx_;

    public:
        Timing start(std::string name)
        {
            std::lock_guard<std::mutex> guard(mtx_);

            Timing timing;
            timing.function = name;
            timing.start = std::clock();

            if (name.length() == 0)
            {
                throw std::invalid_argument("Empty string supplied to 'start'");
            }

            return timing;
        }

        void end(Timing &timing)
        {
            std::lock_guard<std::mutex> guard(mtx_);

            timing.end = std::clock();

            timings_.push_back(timing);

            if (report_.find(timing.function) == report_.end())
            {
                report_[timing.function] = 0;
            }
        }

        friend std::ostream &operator<<(std::ostream &out, Profiler &prof)
        {
            for (auto &t : prof.timings_)
            {
                double duration = 1000.0 * (t.end - t.start) / CLOCKS_PER_SEC;

                prof.report_[t.function] += duration;

                if(t.function.length() == 0)
                {
                    out << "noooo" << std::endl;
                }
            }

            for (const auto &[name, duration] : prof.report_)
            {

                out << "'" << name << "': " << duration << " ms" << std::endl;
            }

            return out;
        }
    };
}

extern cave::Profiler gProfiler;