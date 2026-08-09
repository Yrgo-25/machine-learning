/**
 * @file Activation function types.
 */
#pragma once

#include <cstdint>

namespace ml::act_func
{
/**
 * @brief Enumeration of activation function types.
 */
enum class Type : std::uint8_t
{
    Relu, ///< ReLU (Rectified Linear Unit).
    Tanh, ///< Hyperbolic tangent.
    None, ///< Identity/no activation function.
};
} // namespace ml::act_func
