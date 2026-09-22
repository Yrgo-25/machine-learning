/**
 * @file Real dense layer implementation.
 */
#pragma once

#include <cstddef>

#include "ml/dense_layer/interface.h"
#include "ml/types.h"

namespace ml::dense_layer
{
/**
 * @brief Real dense layer implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Dense final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] nodeCount Number of nodes in the layer. Must be greater than 0.
     * @param[in] weightCount Number of weights per node. Must be greater than 0.
     * @param[in] actFunc Activation function to use (default = ReLU).
     */
    explicit Dense(std::size_t nodeCount, std::size_t weightCount,
                   ActFunc actFunc = ActFunc::Relu) noexcept;

    /**
     * @brief Destructor.
     */
    ~Dense() noexcept override = default;

    /**
     * @brief Get dense layer outputs.
     *
     * @return Matrix holding the dense layer outputs.
     */
    [[nodiscard]] const Matrix1d& output() const noexcept override;

    /**
     * @brief Get computed error values for backpropagation.
     *
     * @return Matrix holding the computed error values.
     */
    [[nodiscard]] const Matrix1d& error() const noexcept override;

    /**
     * @brief Get layer weights for backpropagation.
     *
     * @return Matrix holding the dense layer weights.
     */
    [[nodiscard]] const Matrix2d& weights() const noexcept override;

    /**
     * @brief Get the node count.
     *
     * @return Number of nodes in the layer.
     */
    [[nodiscard]] std::size_t nodeCount() const noexcept override;

    /**
     * @brief Get the weight count of the layer.
     *
     * @return Number of weights per node in the layer.
     */
    [[nodiscard]] std::size_t weightCount() const noexcept override;

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     */
    bool feedforward(const Matrix1d& input) noexcept override;

    /**
     * @brief Perform backpropagation in output layer.
     *
     * @param[in] reference Reference values. Must match the node count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for output layers.
     */
    bool backpropagate(const Matrix1d& reference) noexcept override;

    /**
     * @brief Perform backpropagation in hidden layer.
     *
     * @param[in] nextLayer Next layer of the network.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for hidden layers.
     */
    bool backpropagate(const Interface& nextLayer) noexcept override;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on dimension mismatch or invalid learning rate.
     */
    bool optimize(const Matrix1d& input, double learningRate) noexcept override;

    Dense()                        = delete; // No default constructor.
    Dense(const Dense&)            = delete; // No copy constructor.
    Dense(Dense&&)                 = delete; // No move constructor.
    Dense& operator=(const Dense&) = delete; // No copy assignment.
    Dense& operator=(Dense&&)      = delete; // No move assignment.

private:
    /** Layer weights, holding matrices of weights per node. */
    Matrix2d myWeights;

    /** Layer outputs. */
    Matrix1d myOutput;

    /** Preactivation output. */
    Matrix1d myPreActOutput;

    /** Bias values. */
    Matrix1d myBias;

    /** Computed error values. */
    Matrix1d myError;

    /** Activation function to use. */
    const ActFunc myActFunc;
};
} // namespace ml::dense_layer
