/**
 * @file Machine learning helper functions.
 */
#pragma once

#include "ml/types.h"

namespace ml
{
/**
 * @brief Get the activation function output.
 *
 * @param[in] actFunc Activation function type.
 * @param[in] input Activation function input.
 *
 * @return Corresponding activation function output.
 */
[[nodiscard]] double actFuncOutput(ActFunc actFunc, double input) noexcept;

/**
 * @brief Get the activation function delta.
 *
 * @param[in] actFunc Activation function type.
 * @param[in] input Activation function input.
 *
 * @return Corresponding activation function delta.
 */
[[nodiscard]] double actFuncDelta(ActFunc actFunc, double input) noexcept;

/**
 * @brief Initialize random generator.
 *
 *        This operation is only performed once.
 */
void initRandom() noexcept;

/**
 * @brief Get random floating-point number for initializing a trainable parameter.
 *
 * @return Random floating-point number in range [0.0, 1.0].
 */
double getRandom() noexcept;

} // namespace ml
