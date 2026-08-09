/**
 * @file Random generator interface.
 */
#pragma once

#include <cstdint>

namespace ml::random
{
/**
 * @brief Random generator interface.
 */
class Interface
{
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Generate a random 32-bit integer in the range [0, maxExclusive).
     *
     * @param[in] maxExclusive Upper bound (exclusive) for the random value.
     *
     * @return Random integer in the range [0, maxExclusive).
     */
    [[nodiscard]] virtual std::uint32_t uint32(std::uint32_t maxExclusive) const noexcept = 0;

    /**
     * @brief Generate a random 32-bit integer within the specified range.
     *
     * @param[in] min Minimum value (inclusive).
     * @param[in] max Maximum value (inclusive).
     *
     * @return Random integer in the range [min, max].
     */
    [[nodiscard]] virtual std::int32_t int32(std::int32_t min, std::int32_t max) const noexcept = 0;

    /**
     * @brief Generate a random 64-bit floating point value within the specified range.
     *
     * @param[in] min Minimum value (inclusive).
     * @param[in] max Maximum value (exclusive).
     *
     * @return Random double in the range [min, max).
     */
    [[nodiscard]] virtual double float64(double min, double max) const noexcept = 0;
};
} // namespace ml::random
