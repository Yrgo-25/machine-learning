/**
 * @file Machine learning type definitions.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace ml
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/** Three-dimensional matrix. */
using Matrix3d = std::vector<Matrix2d>;

/** One-dimensional matrix of unsigned integers. */
using MatrixU32 = std::vector<std::uint32_t>;

/**
 * @brief Enumeration of activation functions.
 */
enum class ActFunc : std::uint8_t
{
    Relu, ///< ReLU (Rectified Linear Unit).
    Tanh, ///< Tanh (Hyperbolic tangent).
    None, ///< None (also called Identity).
};
} // namespace ml
