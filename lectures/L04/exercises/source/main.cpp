/**
 * @file Linear regression demo.
 */
#include <cstddef>
#include <cstdio>

#include "ml/lin_reg/fixed.h"
#include "ml/types.h"

namespace
{
// -----------------------------------------------------------------------------
void testModel(const ml::lin_reg::Interface& model, const ml::Matrix1d& inputs) noexcept
{
    std::printf("----------------------------------------\n");
    for (const auto input : inputs)
    {
        const auto output = model.predict(input);
        std::printf("Input: %g, output: %g\n", input, output);
    }
    std::printf("----------------------------------------\n\n");
}
} // namespace

/**
 * @brief Application entry point.
 *
 * @return 0 on successful termination, -1 on error.
 */
int main()
{
    constexpr std::size_t epochCount{20U};
    constexpr double learningRate{0.23};

    // Set up linear regression model predicting y = 2x + 2.
    const ml::Matrix1d trainIn{0.0, 1.0, 2.0, 3.0, 4.0};
    const ml::Matrix1d trainOut{2.0, 4.0, 6.0, 8.0, 10.0};
    ml::lin_reg::Fixed model{trainIn, trainOut};

    // Train the model, print the result on success.
    const auto trained = model.train(epochCount, learningRate);
    if (trained) { testModel(model, trainIn); }
    return trained ? 0 : -1;
}
