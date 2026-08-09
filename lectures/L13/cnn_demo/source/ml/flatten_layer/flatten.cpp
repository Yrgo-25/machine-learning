/**
 * @file Flatten layer implementation details.
 */
#include <cstdio>
#include <exception>

#include "ml/flatten_layer/flatten.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::flatten_layer
{
// -----------------------------------------------------------------------------
Flatten::Flatten(const std::size_t inputSize) noexcept
    : myInputGradients{}
    , myOutput{}
{
    if (0U == inputSize)
    {
        std::fprintf(stderr, "Input size cannot be 0!");
        std::terminate();
    }

    const auto outputSize = inputSize * inputSize;
    initMatrix(myInputGradients, inputSize);
    initMatrix(myOutput, outputSize);
}

// -----------------------------------------------------------------------------
std::size_t Flatten::inputSize() const noexcept { return myInputGradients.size(); }

// -----------------------------------------------------------------------------
std::size_t Flatten::outputSize() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
const Matrix2d& Flatten::inputGradients() const noexcept { return myInputGradients; }

// -----------------------------------------------------------------------------
const Matrix1d& Flatten::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
bool Flatten::feedforward(const Matrix2d& input) noexcept
{
    if (!matchDimensions(myInputGradients.size(), input.size()) || (!isMatrixSquare(input)))
    {
        return false;
    }

    for (std::size_t i{}; i < myInputGradients.size(); ++i)
    {
        for (std::size_t j{}; j < myInputGradients.size(); ++j)
        {
            const auto idx = i * myInputGradients.size() + j;
            myOutput[idx]  = input[i][j];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Flatten::backpropagate(const Matrix1d& outputGradients) noexcept
{
    if (!matchDimensions(myOutput.size(), outputGradients.size())) { return false; }

    for (std::size_t i{}; i < myInputGradients.size(); ++i)
    {
        for (std::size_t j{}; j < myInputGradients.size(); ++j)
        {
            const auto idx         = i * myInputGradients.size() + j;
            myInputGradients[i][j] = outputGradients[idx];
        }
    }
    return true;
}
} // namespace ml::flatten_layer
