/**
 * @file Dense layer interface.
 */
#pragma once

#include <cstddef>

#include "ml/types.h"

namespace ml::dense_layer
{
/**
 * @brief Dense layer interface.
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
     * @brief Get the output values of the layer.
     *
     * @return Matrix holding the output values of the layer.
     */
    [[nodiscard]] virtual const Matrix1d& output() const noexcept = 0;

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    [[nodiscard]] virtual const Matrix1d& inputGradients() const noexcept = 0;

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    virtual bool feedforward(const Matrix1d& input) noexcept = 0;

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    virtual bool backpropagate(const Matrix1d& outputGradients) noexcept = 0;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Matrix holding input data.
     * @param[in] learningRate Learning rate to use.
     *
     * @return True on success, false on failure.
     */
    virtual bool optimize(const Matrix1d& input, double learningRate) noexcept = 0;
};
} // namespace ml::dense_layer
