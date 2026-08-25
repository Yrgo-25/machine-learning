/**
 * @file Linear regression model with fixed learning rate.
 */
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/lin_reg/fixed.h"
#include "ml/types.h"

namespace ml::lin_reg
{
// -----------------------------------------------------------------------------
Fixed::Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept
    : myTrainIn{trainIn}
    , myTrainOut{trainOut}
    , mySetCount{std::min(trainIn.size(), trainOut.size())}
    , myBias{}
    , myWeight{}
{
    if (0U == mySetCount)
    {
        std::fprintf(stderr, "Cannot create regression model without training data!\n");
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
double Fixed::predict(const double input) const noexcept { return myWeight * input + myBias; }

// -----------------------------------------------------------------------------
bool Fixed::train(std::size_t epochCount, double learningRate) noexcept
{
    // Check epoch count, return false if 0.
    if (0U == epochCount) { return false; }

    // Check learning rate, return false if outside range (0.0, 1.0).
    if ((0.0 >= learningRate) || (1.0 <= learningRate)) { return false; }

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        for (std::size_t i{}; i < mySetCount; ++i)
        {
            const auto input  = myTrainIn[i];
            const auto output = myTrainOut[i];
            optimize(input, output, learningRate);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Fixed::optimize(double input, double output, double learningRate) noexcept
{
    // yp = kx + m.
    const auto prediction = predict(input);

    // e  = yref - yp.
    const auto error = output - prediction;

    // m  = m + e * LR
    myBias += error * learningRate;

    // k  = k + e * LR * x
    myWeight += error * learningRate * input;
}
} // namespace ml::lin_reg
