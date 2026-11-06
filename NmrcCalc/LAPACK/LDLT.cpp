#include <cstddef>
#include <iostream>
#include <vector>
#include <span>
#include <cmath>
#include <iomanip>
#include <expected>
#include <string>

/* LDLT decomponetion*/
class LDLT{
    public:
        size_t n;   // synmetric matrix dimension
        std::vector<double> data_;      // n*(n+1)/2 elements for Lij and Dii i>=j

        explicit LDLT(size_t dim) : n(dim), data_(dim*(dim+1)/2, 0.0) {};

        //(i, j) compressed to column-major vector index
        [[nodiscard]] constexpr size_t index(size_t idxR, size_t idxC) const noexcept{
            return (idxC*n - idxC*(idxC-1)/2 + (idxR - idxC));
        }
        // Dii
        [[nodiscard]] double D(size_t idx) const noexcept{
            return data_[index(idx, idx)];
        }
        // Lij: 自动处理对角元 1.0 与上三角 0.0
        [[nodicard]] double L(size_t idxC)

        

};