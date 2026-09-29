/**
 * @file Dense layer implementation.
 */
#pragma once

#include <cstdint>

#include "ml/dense_layer/interface.h"
#include "ml/types.h"

namespace ml::dense_layer
{
/**
 * @brief Dense layer implementation.
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
     * @param[in] weightCount Number of weights per node in the layer. Must be greater than 0.
     * @param[in] actFunc Activation function (default = ReLU).
     */
    explicit Dense(std::uint16_t nodeCount, std::uint16_t weightCount,
                   ActFunc actFunc = ActFunc::Relu) noexcept;

    /**
     * @brief Destructor.
     */
    ~Dense() noexcept override = default;

    /**
     * @brief Get layer output.
     *
     * @return Layer output.
     */
    const Matrix1d& output() const noexcept override;

    /**
     * @brief Get computed error values.
     *
     * @return Computed error values for each node in the layer.
     */
    const Matrix1d& error() const noexcept override;

    /**
     * @brief Get layer bias values.
     *
     * @return Layer bias values.
     */
    const Matrix1d& bias() const noexcept override;

    /**
     * @brief Get layer weights.
     *
     * @return Weights for each node in the layer.
     */
    const Matrix2d& weights() const noexcept override;

    /**
     * @brief Get the node count of this layer.
     *
     * @return Number of nodes in this layer.
     */
    std::uint16_t nodeCount() const noexcept override;

    /**
     * @brief Get the weight count of this layer.
     *
     * @return Number of weights per node in this layer.
     */
    std::uint16_t weightCount() const noexcept override;

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix1d& input) noexcept override;

    /**
     * @brief Perform backpropagation for output layer.
     *
     * @param[in] output Output values. Must match the node count of this layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& output) noexcept override;

    /**
     * @brief Perform backpropagation for hidden layer.
     *
     * @param[in] nextLayer Next layer. Must match the dimensions of this layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Interface& nextLayer) noexcept override;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on failure.
     */
    bool optimize(const Matrix1d& input, double learningRate) noexcept override;

    Dense()                        = delete; // No default constructor.
    Dense(const Dense&)            = delete; // No copy constructor.
    Dense(Dense&&)                 = delete; // No move constructor.
    Dense& operator=(const Dense&) = delete; // No copy assignment.
    Dense& operator=(Dense&&)      = delete; // No move assignment.

private:
    /**
     * @brief Initialize the trainable parameters with random values in range [-1.0, 1.0].
     */
    void initParams() noexcept;

    /** Layer weights, holding one matrix of weights per node. */
    Matrix2d myWeights;

    /** Output values, after the activation function. */
    Matrix1d myOutput;

    /** Output values, before the activation function. */
    Matrix1d myPreActOutput;

    /** Bias values, one per node. */
    Matrix1d myBias;

    /** Computed error values, one per node. */
    Matrix1d myError;

    /** Activation function of the layer. */
    const ActFunc myActFunc;
};
} // namespace ml::dense_layer
