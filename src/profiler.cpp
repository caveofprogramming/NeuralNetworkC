#include "profiler.h"

cave::Profiler gProfiler;

namespace cave
{
    Timing Profiler::start(std::string name)
    {
        Timing timing;

        if (active_)
        {
            std::lock_guard<std::mutex> guard(mtx_);

            timing.function = name;
            timing.start = std::clock();

            if (name.length() == 0)
            {
                throw std::invalid_argument("Empty string supplied to 'start'");
            }
        }
        return timing;
    }

    void Profiler::end(Timing &timing)
    {
        if (active_)
        {
            std::lock_guard<std::mutex> guard(mtx_);

            timing.end = std::clock();

            timings_.push_back(timing);

            if (report_.find(timing.function) == report_.end())
            {
                report_[timing.function] = 0;
            }
        }
    }

    std::ostream &operator<<(std::ostream &out, Profiler &prof)
    {
        if (prof.active_)
        {
            for (auto &t : prof.timings_)
            {
                double duration = 1000.0 * (t.end - t.start) / CLOCKS_PER_SEC;

                prof.report_[t.function] += duration;

                if (t.function.length() == 0)
                {
                    out << "noooo" << std::endl;
                }
            }

            for (const auto &[name, duration] : prof.report_)
            {

                out << "'" << name << "': " << duration << " ms" << std::endl;
            }
        }

        return out;
    }
};
