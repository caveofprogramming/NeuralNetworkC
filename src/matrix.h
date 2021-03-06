#pragma once

#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <fstream>
#include "fileutil.h"

namespace cave
{
    class Matrix: public Serializable
    {
    private:
        int rows_{0};
        int cols_{0};
        std::vector<double> v_;

    public:
        Matrix(){}

        Matrix(int rows, int cols) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);
        }

        Matrix(int rows, int cols, std::vector<double> values, bool rowOrder=true);

        Matrix(int rows, int cols, std::function<double()> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            modify([&](int row, int col, int index, double value)
                   { return init(); });
        }

        Matrix(int rows, int cols, std::function<double(int)> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            modify([&](int row, int col, int index, double value)
                   { return init(index); });
        }

        Matrix(int rows, int cols, std::function<double(int, int, int)> init) : rows_(rows), cols_(cols)
        {
            v_.resize(rows * cols);

            modify([&](int row, int col, int index, double value)
                   { return init(row, col, index); });
        }

        void save(std::ostream &out);
        void load(std::istream &in);

        Matrix transpose() const;
        Matrix colSums();
        Matrix rowMeans();
        Matrix rowSums();
        Matrix largestRowIndexes() const;
        double sum() const;

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
        Matrix apply(std::function<double(int, int, int, double)> f);

        std::string str() const;

        void set(int row, int col, double value);
        void set(int index, double value) { v_[index] = value; }
        double get(int row, int col);
        double get(int index) { return v_[index]; };
        std::vector<double> get() { return v_; };

        double operator[](int index) const
        {
            return v_[index];
        }

        double &operator[](int index)
        {
            return v_[index];
        }

        bool operator==(Matrix const &other);
        bool operator!=(Matrix const &other);

        friend Matrix operator+(Matrix const &m1, Matrix const &m2);
        friend Matrix operator-(Matrix const &m1, Matrix const &m2);
        friend Matrix operator*(Matrix const &m1, Matrix const &m2);
        friend Matrix operator*(double a, Matrix const &m);
        friend Matrix operator-=(Matrix &m1, const Matrix &m2);

        // TODO remove this later.
        Matrix clone() {
            Matrix m(rows_, cols_);
            m.v_ = v_;
            return m;
        };
    };

    std::ostream &operator<<(std::ostream &out, Matrix const &m);
}