#include "matrix.h"
#include <sstream>
#include <iomanip>
#include <assert.h>

namespace cave
{
    Matrix::Matrix(int rows, int cols, std::vector<double> values)
    {
        Matrix m(cols, rows);
        m.v_ = values;
        Matrix transposed = m.transpose();

        *this = std::move(transposed);
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

    Matrix Matrix::sumColumns()
    {
        Matrix result(1, cols_);

        forEach([&](int row, int col, int index, double value)
                { result.v_[col] += value; });

        return result;
    }

    Matrix operator*(double a, Matrix const &m)
    {
        Matrix result(m.rows_, m.cols_);

        for (int i = 0; i < m.v_.size(); ++i)
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
        assert(m1.cols_ == m2.rows_);

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
        assert(m1.rows_ == m2.rows_ && m1.cols_ == m2.cols_);

        return Matrix(m1.rows_, m1.cols_, [&](int index)
                      { return m1[index] + m2[index]; });
    }

    Matrix operator-(const Matrix &m1, const Matrix &m2)
    {
        assert(m1.rows_ == m2.rows_ && m1.cols_ == m2.cols_);

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

    std::string Matrix::str() const
    {
        std::stringstream ss;

        ss << std::fixed;
        ss << std::showpos;

        forEach([&](double row, double col, double value)
                {

            if(col == 0)
            {
                ss << "\n";
            }

            ss << std::setprecision(6);
            ss << std::setw(10);
            ss << value;
            ss << "  "; });

        return ss.str();
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