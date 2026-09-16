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
     * @brief Get dense layer outputs.
     *
     * @return Matrix holding the dense layer outputs.
     */
    [[nodiscard]] virtual const Matrix1d& output() const noexcept = 0;

    /**
     * @brief Get computed error values for backpropagation.
     *
     * @return Matrix holding the computed error values.
     */
    [[nodiscard]] virtual const Matrix1d& error() const noexcept = 0;

    /**
     * @brief Get layer weights for backpropagation.
     *
     * @return Matrix holding the dense layer weights.
     */
    [[nodiscard]] virtual const Matrix2d& weights() const noexcept = 0;

    /**
     * @brief Get the node count.
     *
     * @return Number of nodes in the layer.
     */
    [[nodiscard]] virtual std::size_t nodeCount() const noexcept = 0;

    /**
     * @brief Get the weight count of the layer.
     *
     * @return Number of weights per node in the layer.
     */
    [[nodiscard]] virtual std::size_t weightCount() const noexcept = 0;

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     */
    virtual bool feedforward(const Matrix1d& input) noexcept = 0;

    /**
     * @brief Perform backpropagation in output layer.
     *
     * @param[in] reference Reference values. Must match the node count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for output layers.
     */
    virtual bool backpropagate(const Matrix1d& reference) noexcept = 0;

    /**
     * @brief Perform backpropagation in hidden layer.
     *
     * @param[in] nextLayer Next layer of the network.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for hidden layers.
     */
    virtual bool backpropagate(const Interface& nextLayer) noexcept = 0;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on dimension mismatch or invalid learning rate.
     */
    virtual bool optimize(const Matrix1d& input, double learningRate) noexcept = 0;
};
} // namespace ml::dense_layer
