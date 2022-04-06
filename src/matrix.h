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

        Matrix(int rows, int cols, std::vector<double> values);

        Matrix(int rows, int cols, std::function<double()> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            modify([&](int row, int col, int index, double value){
                return init();
            });
        }

        Matrix(int rows, int cols, std::function<double(int)> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            modify([&](int row, int col, int index, double value){
                return init(index);
            });
        }

        Matrix transpose() const;
        Matrix sumColumns();

        int rows() 
        {
            return rows_;
        }

        int cols() 
        {
            return cols_;
        }

        void forEach(std::function<void(int, int, int, double)> f) const;
        void forEach(std::function<void(int, int, double)> f) const;
        Matrix &modify(std::function<double(int, int, int, double)> f);

        std::string str() const;

        void set(int row, int col, double value);
        double get(int row, int col);
        std::vector<double> get() { return v_; };

        double operator[](int index) const
        {
            return v_[index];
        }

        double &operator[](int index)
        {
            return v_[index];
        }

        friend Matrix operator+(Matrix const &m1, Matrix const &m2);
        friend Matrix operator-(Matrix const &m1, Matrix const &m2);
        friend Matrix operator*(Matrix const &m1, Matrix const &m2);
        friend Matrix operator*(double a, Matrix const &m);
    };

    std::ostream &operator<<(std::ostream &out, Matrix const &m);
}