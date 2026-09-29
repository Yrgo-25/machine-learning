/**
 * @file Dense layer implementation details.
 */
#include <cstdint>
#include <cstdio>
#include <exception>

#include "ml/dense_layer/dense.h"
#include "ml/helpers.h"
#include "ml/types.h"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Dense::Dense(const std::uint16_t nodeCount, const std::uint16_t weightCount,
             const ActFunc actFunc) noexcept
    : myWeights{}
    , myOutput{}
    , myPreActOutput{}
    , myBias{}
    , myError{}
    , myActFunc{actFunc}
{
    if ((0U == nodeCount) || (0U == weightCount))
    {
        std::fprintf(stderr, "Failed to create dense layer: "
                             "node and weight count must both be greater than 0!\n");
        std::terminate();
    }

    // Set the size of all matrices, every value starts at 0.0.
    myWeights.resize(nodeCount, Matrix1d(weightCount));
    myOutput.resize(nodeCount);
    myPreActOutput.resize(nodeCount);
    myBias.resize(nodeCount);
    myError.resize(nodeCount);

    // Initialize trainable parameters.
    initParams();
}

// -----------------------------------------------------------------------------
const Matrix1d& Dense::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const Matrix1d& Dense::error() const noexcept { return myError; }

// -----------------------------------------------------------------------------
const Matrix1d& Dense::bias() const noexcept { return myBias; }

// -----------------------------------------------------------------------------
const Matrix2d& Dense::weights() const noexcept { return myWeights; }

// -----------------------------------------------------------------------------
std::uint16_t Dense::nodeCount() const noexcept
{
    return static_cast<std::uint16_t>(myOutput.size());
}

// -----------------------------------------------------------------------------
std::uint16_t Dense::weightCount() const noexcept
{
    return static_cast<std::uint16_t>(myWeights[0U].size());
}

// -----------------------------------------------------------------------------
bool Dense::feedforward(const Matrix1d& input) noexcept
{
    const auto size     = static_cast<std::uint16_t>(input.size());
    const auto mismatch = size != weightCount();

    if (mismatch)
    {
        std::fprintf(stderr,
                     "Feedforward failed due to dimension mismatch: expected %u, actual = %u!\n",
                     weightCount(), size);
        return false;
    }

    for (std::uint16_t i{}; i < nodeCount(); ++i)
    {
        // Add the bias first.
        auto sum = myBias[i];

        // Add the contribution from the inputs, multiply with the corresponding weights.
        for (std::uint16_t j{}; j < weightCount(); ++j)
        {
            sum += myWeights[0U][j] * input[j];
        }
        // Store the layer output, pre- and post activation function filtering.
        myPreActOutput[i] = sum;
        myOutput[i]       = actFuncOutput(myActFunc, sum);
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Matrix1d& output) noexcept
{
    const auto size     = static_cast<std::uint16_t>(output.size());
    const auto mismatch = size != nodeCount();

    if (mismatch)
    {
        std::fprintf(
            stderr, "Backpropagation failed due to dimension mismatch: expected %u, actual = %u!\n",
            nodeCount(), size);
        return false;
    }

    for (std::uint16_t i{}; i < nodeCount(); ++i)
    {
        // Compute the deviation between the reference value and the predicted output.
        const auto deviation = myOutput[i] - output[i];

        // Scale with the derivative, computed from the weighted sum before activation.
        const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
        myError[i]       = deviation * delta;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const Interface& nextLayer) noexcept
{
    const auto mismatch = nextLayer.weightCount() != nodeCount();

    if (mismatch)
    {
        std::fprintf(
            stderr, "Backpropagation failed due to dimension mismatch: expected %u, actual = %u!\n",
            nodeCount(), nextLayer.weightCount());
        return false;
    }

    for (std::uint16_t i{}; i < nodeCount(); ++i)
    {
        double sum{};

        // Sum the error of each node in the next layer, weighted by the connecting weight.
        for (std::uint16_t j{}; j < nextLayer.nodeCount(); ++j)
        {
            sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        myError[i] = sum;
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    const auto lrValid = ((0.0 < learningRate) && (1.0 > learningRate));

    if (!lrValid)
    {
        std::fprintf(stderr, "Optimization failed due to invalid learning rate %g!\n",
                     learningRate);
        return false;
    }
    const auto size     = static_cast<std::uint16_t>(input.size());
    const auto mismatch = size != weightCount();

    if (mismatch)
    {
        std::fprintf(stderr,
                     "Optimization failed due to dimension mismatch: expected %u, actual = %u!\n",
                     weightCount(), size);
        return false;
    }

    for (std::uint16_t i{}; i < nodeCount(); ++i)
    {
        // Adjust the bias with the full change rate.
        const auto changeRate = myError[i] * learningRate;
        myBias[i] += changeRate;

        // Adjust the weights of the node.
        for (std::uint16_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] += changeRate;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Dense::initParams() noexcept
{
    // Seed the random generator, occurs only for the first layer created.
    initRandom();

    // Randomize the trainable parameters, use range [-1.0, 1.0].
    for (std::uint16_t i{}; i < nodeCount(); ++i)
    {
        myBias[i] = 2.0 * getRandom() - 1.0;

        for (std::uint16_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] = 2.0 * getRandom() - 1.0;
        }
    }
}
} // namespace ml::dense_layer
