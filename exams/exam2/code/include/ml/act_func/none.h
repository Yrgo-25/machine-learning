/**
 * @file Identity activation function implementation (no activation applied).
 */
#pragma once

#include "ml/act_func/interface.h"

namespace ml::act_func
{
/**
 * @brief Identity activation function implementation (returns input unchanged).
 *
 *        This class is non-copyable and non-movable.
 */
class None final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    None() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~None() noexcept override = default;

    /**
     * @brief Compute the activation function output (returns input unchanged).
     *
     * @param[in] input The activation function input.
     *
     * @return The input value (identity function: f(x) = x).
     */
    double output(const double input) const noexcept override { return input; }

    /**
     * @brief Compute the activation function derivative (delta for backpropagation).
     *
     * @param[in] input The activation function input.
     *
     * @return Always returns 1 (derivative of identity function: f'(x) = 1).
     */
    double delta(const double input) const noexcept override
    {
        (void)(input);
        constexpr double deltaVal{1.0};
        return deltaVal;
    }

    None(const None&)            = delete; // No copy constructor.
    None(None&&)                 = delete; // No move constructor.
    None& operator=(const None&) = delete; // No copy assignment.
    None& operator=(None&&)      = delete; // No move assignment.
};
} // namespace ml::act_func
