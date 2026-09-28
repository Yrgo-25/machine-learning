/**
 * @file Real dense layer implementation details.
 */
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/dense_layer/dense.h"
#include "ml/types.h"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Dense::Dense(const std::size_t nodeCount, const std::size_t weightCount,
             const ActFunc actFunc) noexcept
    : myWeights{}
    , myOutput{}
    , myPreActOutput{}
    , myBias{}
    , myError{}
    , myActFunc{actFunc}
{
    if (0U == nodeCount)
    {
        std::fprintf(stderr, "Node count cannot be 0!\n");
        std::terminate();
    }
    if (0U == weightCount)
    {
        std::fprintf(stderr, "Weight count cannot be 0!\n");
        std::terminate();
    }
    myWeights.resize(nodeCount, Matrix1d(weightCount));
    myOutput.resize(nodeCount);
    myPreActOutput.resize(nodeCount);
    myBias.resize(nodeCount);
    myError.resize(nodeCount);
}

// -----------------------------------------------------------------------------
const Matrix1d& Dense::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const Matrix1d& Dense::error() const noexcept { return myError; }

// -----------------------------------------------------------------------------
const Matrix2d& Dense::weights() const noexcept { return myWeights; }

// -----------------------------------------------------------------------------
std::size_t Dense::nodeCount() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
std::size_t Dense::weightCount() const noexcept { return myWeights[0U].size(); }

// -----------------------------------------------------------------------------
bool Dense::feedforward(const Matrix1d& input) noexcept
{
    const auto match = (input.size() == weightCount());

    if (!match)
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
                     weightCount(), input.size());
    }
    return match;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Matrix1d& reference) noexcept
{
    const auto match = reference.size() == nodeCount();

    if (!match)
    {
        std::fprintf(stderr, "Output dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     reference.size());
    }
    return match;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Interface& nextLayer) noexcept
{
    const auto match = nextLayer.weightCount() == nodeCount();

    if (!match)
    {
        std::fprintf(stderr, "Layer dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     nextLayer.weightCount());
    }
    return match;
}

// -----------------------------------------------------------------------------
bool Dense::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    const auto lrValid = ((0.0 < learningRate) && (1.0 > learningRate));

    if (!lrValid)
    {
        std::fprintf(stderr, "Invalid learning rate %g!\n", learningRate);
        return false;
    }
    const auto match = input.size() == weightCount();

    if (!match)
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
                     weightCount(), input.size());
    }
    return match;
}
} // namespace ml::dense_layer
