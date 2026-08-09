/**
 * @file Activation function interface.
 */
#pragma once

namespace ml::act_func
{
/**
 * @brief Activation function interface.
 */
class Interface
{
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Compute the activation function output.
     *
     * @param[in] input The activation function input.
     *
     * @return The activation function value at the given input.
     */
    [[nodiscard]] virtual double output(double input) const noexcept = 0;

    /**
     * @brief Compute the activation function derivative (delta for backpropagation).
     *
     * @param[in] input The activation function input.
     *
     * @return The derivative value at the given input.
     */
    [[nodiscard]] virtual double delta(double input) const noexcept = 0;
};
} // namespace ml::act_func
