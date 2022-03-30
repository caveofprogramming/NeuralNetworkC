#include "matrix.h"
#include <sstream>
#include <iomanip>

namespace cave
{
    std::ostream &operator<<(std::ostream &out, Matrix const &m)
    {
        out << m.str();
        return out;
    }

    Matrix &Matrix::modify(std::function<double(int, int, int, double)> f)
    {
        int index = 0;

        for(int row = 0; row < rows_; ++row) 
        {
            for(int col = 0; col < cols_; col++)
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

        ss << std::fixed << std::showpos;
        ss.precision(5);

        forEach([&](int row, int col, int index, double value){

            if(col == 0)
            {
                ss << "\n";
            }

            ss << value;
            ss << "  ";
        });

        return ss.str();
    }

    void Matrix::forEach(std::function<void(int, int, int, double)> f) const
    {
        int index = 0;

        for(int row = 0; row < rows_; ++row) 
        {
            for(int col = 0; col < cols_; col++)
            {
                f(row, col, index, v_[index]);
                ++index;
            }
        }
    }
}