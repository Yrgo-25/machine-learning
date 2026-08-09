/**
 * @file Flatten layer interface.
 */
#pragma once

#include <cstddef>

#include "ml/types.h"

namespace ml::flatten_layer
{
/**
 * @brief Flatten layer interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Get the input size of the layer.
     *
     * @return The input size of the layer.
     */
    [[nodiscard]] virtual std::size_t inputSize() const noexcept = 0;

    /**
     * @brief Get the output size of the layer.
     *
     * @return The output size of the layer.
     */
    [[nodiscard]] virtual std::size_t outputSize() const noexcept = 0;

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    [[nodiscard]] virtual const Matrix2d& inputGradients() const noexcept = 0;

    /**
     * @brief Get the flattened output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    [[nodiscard]] virtual const Matrix1d& output() const noexcept = 0;

    /**
     * @brief Flatten the input from 2D to 1D.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    virtual bool feedforward(const Matrix2d& input) noexcept = 0;

    /**
     * @brief Unflatten the output gradients from 1D to 2D.
     *
     * @param[in] outputGradients Matrix holding output gradients.
     *
     * @return True on success, false on failure.
     */
    virtual bool backpropagate(const Matrix1d& outputGradients) noexcept = 0;
};
} // namespace ml::flatten_layer
