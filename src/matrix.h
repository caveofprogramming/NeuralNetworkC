#pragma once

#include <vector>
#include <string>
#include <functional>

namespace cave
{
    class Matrix
    {
    private:
        int rows_;
        int cols_;
        std::vector<double> v_;

    public:
        Matrix(int rows, int cols) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);
        }

        Matrix(int rows, int cols, std::function<double()> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            auto modifier = [&](int row, int col, int index, double value){
                return init();
            };

            modify(modifier);
        }

        void forEach(std::function<void(int, int, int, double)> f) const;
        Matrix &modify(std::function<double(int, int, int, double)> f);

        std::string str() const;
    };

    std::ostream &operator<<(std::ostream &out, Matrix const &m);
}