/**
 * @file Tanh (hyperbolic tangent) activation function implementation.
 */
#pragma once

#include <cmath>

#include "ml/act_func/interface.h"

namespace ml::act_func
{
/**
 * @brief Tanh (hyperbolic tangent) activation function implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Tanh final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    Tanh() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Tanh() noexcept override = default;

    /**
     * @brief Compute the activation function output.
     *
     * @param[in] input The activation function input.
     *
     * @return Hyperbolic tangent of input (tanh function: f(x) = tanh(x), range [-1, 1]).
     */
    double output(const double input) const noexcept override { return std::tanh(input); }

    /**
     * @brief Compute the activation function derivative (delta for backpropagation).
     *
     * @param[in] input The activation function input.
     *
     * @return Derivative of tanh (f'(x) = 1 - tanh²(x)).
     */
    double delta(const double input) const noexcept override
    {
        const double out{std::tanh(input)};
        return 1.0 - out * out;
    }

    Tanh(const Tanh&)            = delete; // No copy constructor.
    Tanh(Tanh&&)                 = delete; // No move constructor.
    Tanh& operator=(const Tanh&) = delete; // No copy assignment.
    Tanh& operator=(Tanh&&)      = delete; // No move assignment.
};
} // namespace ml::act_func
