/**
 * @file Linear regression model with fixed learning rate.
 */
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>

#include "ml/lin_reg/fixed.h"
#include "ml/types.h"

namespace ml::lin_reg
{
namespace
{
// -----------------------------------------------------------------------------
void initRandom() noexcept
{
    // Only initialize the random generator once.
    static bool initialized{false};
    if (initialized) { return; }

    // Initialize the random generator with the current timestamp as seed.
    std::srand(std::time(nullptr));
    initialized = true;
}
} // namespace

// -----------------------------------------------------------------------------
Fixed::Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept
    : myTrainOrder{}
    , myTrainIn{trainIn}
    , myTrainOut{trainOut}
    , myBias{}
    , myWeight{}
{
    const auto setCount = std::min(trainIn.size(), trainOut.size());

    if (0U == setCount)
    {
        std::fprintf(stderr, "Cannot create regression model without training data!\n");
        std::terminate();
    }

    // Initialize training order vector with indexes of the training sets.
    myTrainOrder.resize(setCount);

    for (std::uint32_t i{}; i < setCount; ++i)
    {
        myTrainOrder[i] = i;
    }

    // Initialize random generator (occurs only once).
    initRandom();
}

// -----------------------------------------------------------------------------
double Fixed::predict(const double input) const noexcept { return myWeight * input + myBias; }

// -----------------------------------------------------------------------------
bool Fixed::train(const std::size_t epochCount, const double learningRate,
                  const double precisionThreshold) noexcept
{
    constexpr std::size_t evaluationInterval{5U};

    // Check epoch count, return false if 0.
    if (0U == epochCount) { return false; }

    // Check learning rate, return false if outside range (0.0, 1.0).
    if ((0.0 >= learningRate) || (1.0 <= learningRate)) { return false; }

    // Check precision threshold, return false if outside range (0.0, 1.0).
    if ((0.0 >= precisionThreshold) || (1.0 <= precisionThreshold)) { return false; }

    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        shuffle();

        for (const auto i : myTrainOrder)
        {
            const auto input  = myTrainIn[i];
            const auto output = myTrainOut[i];
            optimize(input, output, learningRate);
        }

        // Evaluate the precision every tenth epoch, skip the first one.
        // Note: epoch % evaluationInterval = epoch % 10 => rest of epoch / 10 = 0 when
        // epoch = 0, 10, 20, 30, 40, 50.
        const auto evaluate = ((0U < epoch) && (0U == (epoch % evaluationInterval)));

        if (evaluate)
        {
            // Compute precision, stop training and print result if above the given threshold.
            const auto precision = computePrecision();

            if (precision >= precisionThreshold)
            {
                std::printf("Finished training with precision %g after %zu epochs!\n", precision,
                            epoch);
                return true;
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Fixed::optimize(const double input, const double output, const double learningRate) noexcept
{
    // m == yref if x == 0.
    if (0.0 == input)
    {
        myBias = output;
        return;
    }

    const auto prediction = predict(input);      // yp = kx + m.
    const auto error      = output - prediction; // e  = yref - yp.
    myBias += error * learningRate;              // m  = m + e * LR
    myWeight += error * learningRate * input;    // k  = k + e * LR * x
}

// -----------------------------------------------------------------------------
void Fixed::shuffle() noexcept
{
    // Iterate through all training sets, swap each index i with a random index r.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto r    = std::rand() % myTrainOrder.size();
        const auto temp = myTrainOrder[i];
        myTrainOrder[i] = myTrainOrder[r];
        myTrainOrder[r] = temp;
    }
}

// -----------------------------------------------------------------------------
double Fixed::computePrecision() const noexcept
{
    double sum{};

    // Iterate through all training sets.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto input      = myTrainIn[i];
        const auto output     = myTrainOut[i];
        const auto prediction = predict(input);

        // Compute the current error (absolute value, use std::abs() from <cmath>).
        const auto error = std::abs(output - prediction);
        sum += error; // sum = sum + error
    }
    // Compute average error, then return 1.0 - the average.
    // The return value will be very close to 1.0 if the model predicts well.
    const auto avgError = sum / myTrainOrder.size();
    return 1.0 - avgError;
}
} // namespace ml::lin_reg
