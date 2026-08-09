/**
 * @file Machine learning utility functions.
 */
#include <cstddef>
#include <cstdio>

#include "ml/random/generator.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml
{
// -----------------------------------------------------------------------------
void initMatrix(Matrix1d& matrix) noexcept
{
    for (auto& num : matrix)
    {
        num = 0.0;
    }
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix) noexcept
{
    for (auto& row : matrix)
    {
        for (auto& column : row)
        {
            column = 0.0;
        }
    }
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix1d& matrix, const std::size_t size) noexcept
{
    matrix.resize(size);
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix, const std::size_t size) noexcept
{
    matrix.resize(size, Matrix1d(size));
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix, const std::size_t rowCount, const std::size_t colCount) noexcept
{
    matrix.resize(rowCount, Matrix1d(colCount));
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
bool isMatrixSquare(const Matrix2d& matrix) noexcept
{
    for (std::size_t i{}; i < matrix.size(); ++i)
    {
        if (matrix[i].size() != matrix.size()) { return false; }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool isMatrixSquare(const Matrix2d& matrix, const std::size_t row) noexcept
{
    return matrix.size() <= row ? false : matrix[row].size() == matrix.size();
}

// -----------------------------------------------------------------------------
void printMatrix(const Matrix2d& matrix) noexcept
{
    std::printf("[");
    const auto* lastColumn{matrix.empty() ? nullptr : &matrix[matrix.size() - 1U]};

    for (const auto& column : matrix)
    {
        std::printf("[");
        const auto* lastRow{column.empty() ? nullptr : &column[column.size() - 1U]};

        for (const auto& row : column)
        {
            std::printf("%.1f", row);
            if (&row < lastRow) { std::printf(", "); }
        }
        if (&column < lastColumn) { std::printf("],\n"); }
        else { std::printf("]"); }
    }
    std::printf("]\n");
}

// -----------------------------------------------------------------------------
void printMatrix(const Matrix1d& matrix) noexcept
{
    std::printf("[");
    const auto* lastNum{matrix.empty() ? nullptr : &matrix[matrix.size() - 1U]};

    for (const auto& num : matrix)
    {
        std::printf("%.1f", num);
        if (&num < lastNum) { std::printf(", "); }
    }
    std::printf("]\n");
}

// -----------------------------------------------------------------------------
bool matchDimensions(const std::size_t expectedSize, const std::size_t actualSize) noexcept
{
    return expectedSize == actualSize;
}

// -----------------------------------------------------------------------------
bool checkLearningRate(const double learningRate) noexcept
{
    return (0.0 < learningRate) && (1.0 > learningRate);
}

// -----------------------------------------------------------------------------
double randomStartVal() noexcept
{
    constexpr double min{0.0};
    constexpr double max{1.0};
    return random::Generator::getInstance().float64(min, max);
}

// -----------------------------------------------------------------------------
TrainOrderList createTrainOrderList(const std::size_t trainSetCount) noexcept
{
    TrainOrderList list(trainSetCount);

    for (std::size_t i{}; i < trainSetCount; ++i)
    {
        list[i] = i;
    }
    return list;
}

// -----------------------------------------------------------------------------
void shuffleTrainOrderList(TrainOrderList& list) noexcept
{
    for (std::size_t i{}; i < list.size(); ++i)
    {
        const auto r{random::Generator::getInstance().uint32(list.size())};
        const auto temp{list[i]};
        list[i] = list[r];
        list[r] = temp;
    }
}
} // namespace ml
