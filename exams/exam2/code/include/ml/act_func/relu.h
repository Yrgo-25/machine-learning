/**
 * @file ReLU (Rectified Linear Unit) activation function implementation.
 */
#pragma once

#include "ml/act_func/interface.h"

namespace ml::act_func
{
/**
 * @brief ReLU (Rectified Linear Unit) activation function implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Relu final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    Relu() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Relu() noexcept override = default;

    /**
     * @brief Compute the activation function output.
     *
     * @param[in] input The activation function input.
     *
     * @return The input if positive, otherwise 0 (ReLU function: f(x) = max(0, x)).
     */
    double output(const double input) const noexcept override { return 0.0 < input ? input : 0.0; }

    /**
     * @brief Compute the activation function derivative (delta for backpropagation).
     *
     * @param[in] input The activation function input.
     *
     * @return 1 if input is positive, otherwise 0 (ReLU derivative: f'(x) = 1 if x > 0, else 0).
     */
    double delta(const double input) const noexcept override { return 0.0 < input ? 1.0 : 0.0; }

    Relu(const Relu&)            = delete; // No copy constructor.
    Relu(Relu&&)                 = delete; // No move constructor.
    Relu& operator=(const Relu&) = delete; // No copy assignment.
    Relu& operator=(Relu&&)      = delete; // No move assignment.
};
} // namespace ml::act_func
