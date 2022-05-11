#include "matrix.h"
#include <sstream>
#include <iomanip>
#include <assert.h>
#include <exception>
#include <cmath>

namespace cave
{
    Matrix::Matrix(int rows, int cols, std::vector<double> values, bool rowOrder)
    {
        if (rowOrder)
        {
            v_ = values;
        }
        else
        {
            Matrix m(cols, rows);
            m.v_ = values;
            Matrix transposed = m.transpose();

            v_ = transposed.v_;
        }

        rows_ = rows;
        cols_ = cols;
    }

    bool Matrix::operator!=(Matrix const &other)
    {
        return !(*this == other);
    }

    bool Matrix::operator==(Matrix const &other)
    {
        const double tolerance = 0.01;

        for (std::size_t i = 0; i < v_.size(); ++i)
        {
            double value1 = v_[i];
            double value2 = other.v_[i];

            if (abs(value2 - value1) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    std::ostream &operator<<(std::ostream &out, Matrix const &m)
    {
        out << m.str();
        return out;
    }

    void Matrix::set(int row, int col, double value)
    {
        v_[row * cols_ + col] = value;
    }

    double Matrix::get(int row, int col)
    {
        return v_[row * cols_ + col];
    }

    Matrix Matrix::rowMeans()
    {
        Matrix result(rows_, 1);

        forEach([&](int row, int col, int index, double value)
                { result.v_[row] += value / cols_; });

        return result;
    }

    Matrix Matrix::rowSums()
    {
        Matrix result(rows_, 1);

        forEach([&](int row, int col, int index, double value)
                { result.v_[row] += value; });

        return result;
    }

    Matrix Matrix::colSums()
    {
        Matrix result(1, cols_);

        forEach([&](int row, int col, int index, double value)
                { result.v_[col] += value; });

        return result;
    }

    Matrix operator*(double a, Matrix const &m)
    {
        Matrix result(m.rows_, m.cols_);

        for (std::size_t i = 0; i < m.v_.size(); ++i)
        {
            result.v_[i] = m.v_[i] * a;
        }

        return result;
    }

    Matrix Matrix::transpose() const
    {
        Matrix result(cols_, rows_);

        forEach([&](int row, int col, int index, double value)
                { result[col * rows_ + row] = value; });

        return result;
    }

    Matrix operator*(const Matrix &m1, const Matrix &m2)
    {
        if (m1.cols_ != m2.rows_)
        {
            std::stringstream ss;
            ss << "Matrixes cannot be multiplied: ";
            ss << m1.rows_ << "x" << m1.cols_ << " * " << m2.rows_ << "x" << m2.cols_ << std::endl;
            throw std::logic_error(ss.str());
        }

        Matrix result(m1.rows_, m2.cols_);

        int index = 0;

        for (int row = 0; row < result.rows_; ++row)
        {
            for (int col = 0; col < result.cols_; ++col)
            {
                for (int n = 0; n < m1.cols_; ++n)
                {
                    result.v_[index] += m1.v_[row * m1.cols_ + n] * m2.v_[col + n * m2.cols_];
                }
                ++index;
            }
        }

        return result;
    }

    Matrix operator+(const Matrix &m1, const Matrix &m2)
    {
        assert(m1.rows_ == m2.rows_ && m1.cols_ == m2.cols_ && "Matrix addition failed.");

        return Matrix(m1.rows_, m1.cols_, [&](int index)
                      { return m1[index] + m2[index]; });
    }

    Matrix operator-(const Matrix &m1, const Matrix &m2)
    {
        assert(m1.rows_ == m2.rows_ && m1.cols_ == m2.cols_ && "Matrix subtraction failed");

        return Matrix(m1.rows_, m1.cols_, [&](int index)
                      { return m1[index] - m2[index]; });
    }

    Matrix &Matrix::modify(std::function<double(int, int, int, double)> f)
    {
        int index = 0;

        for (int row = 0; row < rows_; ++row)
        {
            for (int col = 0; col < cols_; col++)
            {
                v_[index] = f(row, col, index, v_[index]);
                ++index;
            }
        }

        return *this;
    }

    Matrix Matrix::apply(std::function<double(int, int, int, double)> f)
    {
        Matrix result;
        result.v_ = v_;
        result.rows_ = rows_;
        result.cols_ = cols_;

        result.modify(f);

        return result;
    }

    std::string Matrix::str() const
    {
        std::stringstream ss;
        const int maxRows = 8;
        const int maxCols = 8;

        if(rows_ >= maxRows || cols_ >= maxCols)
        {
            ss << "\n[truncated from " << rows_ << "x" << cols_ << "]";
        }

        ss << std::fixed;
        ss << std::showpos;

        // clang-format off
        forEach([&](double row, double col, double value)
        {
            if(col >= maxCols || row >= maxRows)
            {
                return;
            }

            if(col == 0)
            {
                ss << "\n";
            }
           
            ss << std::setprecision(6);
            ss << std::setw(12);
            ss << value;
            ss << "  "; 
        });
        // clang-format on

        ss << "\n\n\n";

        return ss.str();
    }

    double Matrix::sum() const
    {
        double total = 0;

        for (auto &v : v_)
        {
            total += v;
        }

        return total;
    }

    Matrix Matrix::largestRowIndexes() const
    {
        Matrix result(1, cols_);

        std::vector<double> largest(cols_);

        // clang-format off
        forEach([&](int row, int col, int index, double value)
        { 
            if(value > largest[col])
            {
                largest[col] = value;
                result.v_[col] = row;
            }
        });
        // clang-format on

        return result;
    }

    void Matrix::forEach(std::function<void(int, int, double)> f) const
    {
        forEach([&](int row, int col, int index, double value)
                { f(row, col, value); });
    }

    void Matrix::forEach(std::function<void(int, int, int, double)> f) const
    {
        int index = 0;

        for (int row = 0; row < rows_; ++row)
        {
            for (int col = 0; col < cols_; col++)
            {
                f(row, col, index, v_[index]);
                ++index;
            }
        }
    }
}