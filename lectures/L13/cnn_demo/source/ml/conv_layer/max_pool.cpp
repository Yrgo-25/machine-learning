/**
 * @file Max pooling layer implementation details.
 */
#include <cstdio>
#include <exception>

#include "ml/conv_layer/max_pool.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
namespace
{
// -----------------------------------------------------------------------------
bool isInRange(const Matrix2d& matrix, const std::size_t col, const std::size_t row) noexcept
{
    return (matrix.size() > col) && (matrix[col].size() > row);
}
} // namespace

// -----------------------------------------------------------------------------
MaxPool::MaxPool(const std::size_t inputSize, const std::size_t poolSize) noexcept
    : myInput{}
    , myInputGradients{}
    , myOutput{}
{
    checkDimensions(inputSize, poolSize);

    const auto outputSize = inputSize / poolSize;
    initMatrix(myInput, inputSize);
    initMatrix(myInputGradients, inputSize);
    initMatrix(myOutput, outputSize);
}

// -----------------------------------------------------------------------------
std::size_t MaxPool::inputSize() const noexcept { return myInputGradients.size(); }

// -----------------------------------------------------------------------------
std::size_t MaxPool::outputSize() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
const Matrix2d& MaxPool::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const Matrix2d& MaxPool::inputGradients() const noexcept { return myInputGradients; }

// -----------------------------------------------------------------------------
bool MaxPool::feedforward(const Matrix2d& input) noexcept
{
    if (!matchDimensions(myInput.size(), input.size()) || (!isMatrixSquare(input)))
    {
        return false;
    }

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            const auto blockCol = i * poolSize();
            const auto blockRow = j * poolSize();
            auto maxVal         = input[blockCol][blockRow];

            for (std::size_t pi{}; pi < poolSize(); ++pi)
            {
                for (std::size_t pj{}; pj < poolSize(); ++pj)
                {
                    const auto col = blockCol + pi;
                    const auto row = blockRow + pj;

                    if (isInRange(input, col, row))
                    {
                        auto val = input[col][row];
                        if (maxVal < val) { maxVal = val; }
                    }
                }
            }
            myOutput[i][j] = maxVal;
        }
    }
    myInput = input;
    return true;
}

// -----------------------------------------------------------------------------
bool MaxPool::backpropagate(const Matrix2d& outputGradients) noexcept
{
    if (!matchDimensions(myOutput.size(), outputGradients.size())) { return false; }
    initMatrix(myInputGradients);

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        if (!isMatrixSquare(outputGradients, i)) { return false; }

        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            const auto blockCol = i * poolSize();
            const auto blockRow = j * poolSize();

            auto maxVal = myInput[blockCol][blockRow];
            auto maxCol = blockCol;
            auto maxRow = blockRow;

            for (std::size_t pi{}; pi < poolSize(); ++pi)
            {
                for (std::size_t pj{}; pj < poolSize(); ++pj)
                {
                    auto col = blockCol + pi;
                    auto row = blockRow + pj;

                    if (isInRange(myInput, col, row))
                    {
                        const auto val = myInput[col][row];

                        if (maxVal < val)
                        {
                            maxCol = col;
                            maxRow = row;
                            maxVal = val;
                        }
                    }
                }
            }
            myInputGradients[maxCol][maxRow] += outputGradients[i][j];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool MaxPool::optimize(const double learningRate) noexcept
{
    // No optimization is required for pooling layers since they have no trainable parameters.
    // This method is implemented to satisfy the common layer interface.
    (void)(learningRate);
    return true;
}

// -----------------------------------------------------------------------------
std::size_t MaxPool::poolSize() const noexcept { return myInput.size() / myOutput.size(); }

// -----------------------------------------------------------------------------
void MaxPool::checkDimensions(const std::size_t inputSize,
                              const std::size_t poolSize) const noexcept
{
    if (0U == inputSize)
    {
        std::fprintf(stderr, "Input size cannot be 0!");
        std::terminate();
    }
    else if (0U == poolSize)
    {
        std::fprintf(stderr, "Pool size cannot be 0!");
        std::terminate();
    }
    else if (inputSize < poolSize)
    {
        std::fprintf(stderr, "Input size cannot be smaller than the pool size!");
        std::terminate();
    }
    else if (0U != (inputSize % poolSize))
    {
        std::fprintf(stderr, "Input size must be divisible by pool size!");
        std::terminate();
    }
}
} // namespace ml::conv_layer
