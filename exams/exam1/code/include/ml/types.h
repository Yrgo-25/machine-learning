/**
 * @file Machine learning definitions.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace ml
{
/**
 * @brief Enumeration of activation functions.
 */
enum class ActFunc : std::uint8_t
{
    Relu, ///< Rectified Linear Unit.
    Tanh, ///< Hyperbolic tangent.
    None, ///< Identity (passthrough).
};

// clang-format off

/** Dense layer interface. */
namespace dense_layer { class Interface; }

/** Neural network interface. */
namespace neural_network { class Interface; }

// clang-format on

/** Dense layer interface. */
using IDenseLayer = dense_layer::Interface;

/** Neural network interface. */
using INeuralNetwork = neural_network::Interface;

/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/** Three-dimensional matrix. */
using Matrix3d = std::vector<Matrix2d>;

} // namespace ml
