/**
 * @file Dense layer interface.
 */
#pragma once

#include <cstdint>

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
     * @brief Get layer output.
     *
     * @return Layer output.
     */
    [[nodiscard]] virtual const Matrix1d& output() const noexcept = 0;

    /**
     * @brief Get computed error values.
     *
     * @return Computed error values for each node in the layer.
     */
    [[nodiscard]] virtual const Matrix1d& error() const noexcept = 0;

    /**
     * @brief Get layer weights.
     *
     * @return Weights for each node in the layer.
     */
    [[nodiscard]] virtual const Matrix2d& weights() const noexcept = 0;

    /**
     * @brief Get the node count of this layer.
     *
     * @return Number of nodes in this layer.
     */
    [[nodiscard]] virtual std::uint16_t nodeCount() const noexcept = 0;

    /**
     * @brief Get the weight count of this layer.
     *
     * @return Number of weights per node in this layer.
     */
    [[nodiscard]] virtual std::uint16_t weightCount() const noexcept = 0;

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     *
     * @return True on success, false on failure.
     */
    virtual bool feedforward(const Matrix1d& input) noexcept = 0;

    /**
     * @brief Perform backpropagation for output layer.
     *
     * @param[in] output Output values. Must match the node count of this layer.
     *
     * @return True on success, false on failure.
     */
    virtual bool backpropagate(const Matrix1d& output) noexcept = 0;

    /**
     * @brief Perform backpropagation for hidden layer.
     *
     * @param[in] nextLayer Next layer. Must match the dimensions of this layer.
     *
     * @return True on success, false on failure.
     */
    virtual bool backpropagate(const Interface& nextLayer) noexcept = 0;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on failure.
     */
    virtual bool optimize(const Matrix1d& input, double learningRate) noexcept = 0;
};
} // namespace ml::dense_layer
