/**
 * @file Real dense layer implementation details.
 */
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>

#include "ml/dense_layer/dense.h"
#include "ml/types.h"

namespace ml::dense_layer
{
namespace
{
// -----------------------------------------------------------------------------
void initRandom() noexcept
{
    static bool initialized{false};

    // Skip initialization if already done.
    if (initialized) { return; }

    // Initialize the random generator, use current time as seed.
    std::srand(std::time(nullptr));
    initialized = true;
}

// -----------------------------------------------------------------------------
[[nodiscard]] double randomStartVal() noexcept
{
    constexpr double max{static_cast<double>(RAND_MAX)}; // 65535.0
    const auto ratio = rand() / max;                     // 0.0 - 1.0.
    return 2.0 * ratio - 1.0;                            // -1.0 - 1.0.
}

// -----------------------------------------------------------------------------
[[nodiscard]] double actFuncOutput(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
            return 0.0 < input ? input : 0.0;
        case ActFunc::Tanh:
            return std::tanh(input);
        default:
            return input;
    }
}

// -----------------------------------------------------------------------------
[[nodiscard]] double actFuncDelta(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
        {
            return 0.0 < input ? 1.0 : 0.0;
        }
        case ActFunc::Tanh:
        {
            const auto out = std::tanh(input);
            return 1.0 - out * out;
        }
        default:
        {
            return 1.0;
        }
    }
}
} // namespace

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

    // Set size of matrices, all values start at 0.0.
    myWeights.resize(nodeCount, Matrix1d(weightCount));
    myOutput.resize(nodeCount);
    myPreActOutput.resize(nodeCount);
    myBias.resize(nodeCount);
    myError.resize(nodeCount);

    // Randomize trainable parameters, use range [-1.0, 1.0].
    initRandom();

    for (std::size_t i{}; i < nodeCount; ++i)
    {
        myBias[i] = randomStartVal();

        for (std::size_t j{}; j < weightCount; ++j)
        {
            myWeights[i][j] = randomStartVal();
        }
    }
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
        return false;
    }

    // Compute new outputs for each node in the layer, one by one.
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        // Add the bias first.
        auto sum = myBias[i];

        // Add the contribution from the inputs, multiply with the corresponding weights.
        for (std::size_t j{}; j < weightCount(); ++j)
        {
            sum += myWeights[i][j] * input[j];
        }
        // Store the layer output, pre- and post activation function filtering.
        myPreActOutput[i] = sum;
        myOutput[i]       = actFuncOutput(myActFunc, sum);
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Matrix1d& reference) noexcept
{
    const auto match = reference.size() == nodeCount();

    if (!match)
    {
        std::fprintf(stderr, "Output dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     reference.size());
        return false;
    }

    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        const auto error = reference[i] - myOutput[i];
        const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
        myError[i]       = error * delta;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Interface& nextLayer) noexcept
{
    const auto match = nextLayer.weightCount() == nodeCount();

    if (!match)
    {
        std::fprintf(stderr, "Layer dimension mismatch: expected %zu, actual: %zu!\n", nodeCount(),
                     nextLayer.weightCount());
        return false;
    }

    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        double sum{};

        for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)
        {
            sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
        myError[i]       = sum * delta;
    }
    return true;
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
        return false;
    }

    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        const auto changeRate = myError[i] * learningRate;
        myBias[i] += changeRate;

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] += changeRate * input[j];
        }
    }
    return true;
}
} // namespace ml::dense_layer
