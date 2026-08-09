/**
 * @file Random generator implementation.
 */
#pragma once

#include <cstdint>

#include "ml/random/interface.h"

namespace ml::random
{
/**
 * @brief Random generator implementation.
 *
 *        This class uses the singleton pattern and is non-copyable and non-movable.
 */
class Generator final : public Interface
{
public:
    /**
     * @brief Get singleton random generator instance.
     *
     * @return Reference to the singleton random generator instance.
     */
    static Interface& getInstance() noexcept;

    /**
     * @brief Generate a random 32-bit integer in the range [0, maxExclusive).
     *
     * @param[in] maxExclusive Upper bound (exclusive) for the random value.
     *
     * @return Random integer in the range [0, maxExclusive).
     */
    std::uint32_t uint32(std::uint32_t maxExclusive) const noexcept override;

    /**
     * @brief Generate a random 32-bit integer within the specified range.
     *
     * @param[in] min Minimum value (inclusive).
     * @param[in] max Maximum value (inclusive).
     *
     * @return Random integer in the range [min, max].
     */
    std::int32_t int32(std::int32_t min, std::int32_t max) const noexcept override;

    /**
     * @brief Generate a random 64-bit floating point value within the specified range.
     *
     * @param[in] min Minimum value (inclusive).
     * @param[in] max Maximum value (exclusive).
     *
     * @return Random double in the range [min, max).
     */
    double float64(double min, double max) const noexcept override;

    Generator(const Generator&)            = delete; // No copy constructor.
    Generator(Generator&&)                 = delete; // No move constructor.
    Generator& operator=(const Generator&) = delete; // No copy assignment.
    Generator& operator=(Generator&&)      = delete; // No move assignment.

private:
    Generator() noexcept;
    ~Generator() noexcept override = default;
};
} // namespace ml::random
