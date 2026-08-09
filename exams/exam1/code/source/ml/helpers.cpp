/**
 * @file Machine learning helper function definitions.
 */
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "ml/helpers.h"
#include "ml/types.h"

namespace ml
{
// --------------------------------------------------------------------------------
double actFuncOutput(const ActFunc actFunc, const double input) noexcept
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

// --------------------------------------------------------------------------------
double actFuncDelta(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
            return 0.0 < input ? 1.0 : 0.0;
        case ActFunc::Tanh:
            return 1.0 - std::tanh(input) * std::tanh(input);
        default:
            return 1.0;
    }
}

// --------------------------------------------------------------------------------
void initRandom() noexcept
{
    static bool initialized{false};
    if (initialized) { return; }
    std::srand(std::time(nullptr));
    initialized = true;
}

// --------------------------------------------------------------------------------
double getRandom() noexcept
{
    constexpr auto max = static_cast<double>(RAND_MAX);
    return std::rand() / max;
}
} // namespace ml
