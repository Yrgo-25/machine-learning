/**
 * @file Convolutional layer interface.
 */
#pragma once

#include <cstddef>
#include <memory>

#include "ml/types.h"

namespace ml::conv_layer
{
/**
 * @brief Convolutional layer interface.
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
     * @brief Get the output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    [[nodiscard]] virtual const Matrix2d& output() const noexcept = 0;

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    [[nodiscard]] virtual const Matrix2d& inputGradients() const noexcept = 0;

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    virtual bool feedforward(const Matrix2d& input) noexcept = 0;

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    virtual bool backpropagate(const Matrix2d& outputGradients) noexcept = 0;

    /**
     * @brief Perform optimization.
     *
     * @param[in] learningRate Learning rate to use.
     *
     * @return True on success, false on failure.
     */
    virtual bool optimize(double learningRate) noexcept = 0;
};
} // namespace ml::conv_layer
