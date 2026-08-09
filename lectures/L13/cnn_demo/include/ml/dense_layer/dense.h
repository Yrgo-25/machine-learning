/**
 * @file Dense layer implementation.
 */
#pragma once

#include <cstddef>

#include "ml/act_func/type.h"
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
     * @brief Create a new dense layer.
     *
     * @param[in] inputSize Input size.
     * @param[in] outputSize Output size.
     * @param[in] actFunc Activation function to use for this layer (default = ReLU).
     */
    explicit Dense(std::size_t inputSize, std::size_t outputSize,
                   act_func::Type actFunc = act_func::Type::Relu) noexcept;

    /**
     * @brief Destructor.
     */
    ~Dense() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     *
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override;

    /**
     * @brief Get the output size of the layer.
     *
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override;

    /**
     * @brief Get the output values of the layer.
     *
     * @return Matrix holding the output values of the layer.
     */
    const Matrix1d& output() const noexcept override;

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix1d& inputGradients() const noexcept override;

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix1d& input) noexcept override;

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override;

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Matrix holding input data.
     * @param[in] learningRate Learning rate to use.
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
    void checkParameters(std::size_t inputSize, std::size_t outputSize) noexcept;
    void init(std::size_t inputSize, std::size_t outputSize, act_func::Type actFunc) noexcept;

    /** Input gradients. */
    Matrix1d myInputGradients;

    /** Bias values. */
    Matrix1d myBias;

    /** Weights for each node. */
    Matrix2d myWeights;

    /** Output matrix. */
    Matrix1d myOutput;

    /** Pre-activation output matrix. */
    Matrix1d myPreActivationOutput;

    /** Error values. */
    Matrix1d myError;

    /** Activation function. */
    ActFuncPtr myActFunc;
};
} // namespace ml::dense_layer
