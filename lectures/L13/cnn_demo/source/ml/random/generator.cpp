/**
 * @file Random generator implementation details.
 */
#include <cstdint>
#include <cstdlib>
#include <ctime>

#include "ml/random/generator.h"

namespace ml::random
{
// -----------------------------------------------------------------------------
Interface& Generator::getInstance() noexcept
{
    static Generator myInstance{};
    return myInstance;
}

// -----------------------------------------------------------------------------
std::uint32_t Generator::uint32(const std::uint32_t maxExclusive) const noexcept
{
    return static_cast<std::uint32_t>(std::rand() % maxExclusive);
}

// -----------------------------------------------------------------------------
std::int32_t Generator::int32(const std::int32_t min, const std::int32_t max) const noexcept
{
    if (min >= max) { return min; }
    return std::rand() % (max - min + 1) + min;
}

// -----------------------------------------------------------------------------
double Generator::float64(const double min, const double max) const noexcept
{
    if (min >= max) { return min; }
    return (std::rand() / static_cast<double>(RAND_MAX)) * (max - min) + min;
}

// -----------------------------------------------------------------------------
Generator::Generator() noexcept { std::srand(std::time(nullptr)); }
} // namespace ml::random
