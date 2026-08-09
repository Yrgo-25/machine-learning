/**
 * @file Max pooling layer implementation.
 */
#pragma once

#include "ml/conv_layer/interface.h"
#include "ml/types.h"

namespace ml::conv_layer
{
/**
 * @brief Max pooling layer implementation.
 */
class MaxPool final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     */
    explicit MaxPool(std::size_t inputSize, std::size_t poolSize) noexcept;

    /**
     * @brief Default destructor.
     */
    ~MaxPool() noexcept override = default;

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
     * @brief Get the output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override;

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override;

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override;

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override;

    /**
     * @brief Perform optimization (not implemented for pooling layers).
     *
     * @param[in] learningRate Learning rate to use.
     *
     * @return True (optimization is a no-op for pooling layers).
     */
    bool optimize(double learningRate) noexcept override;

    MaxPool()                          = delete; // No default constructor.
    MaxPool(const MaxPool&)            = delete; // No copy constructor.
    MaxPool(MaxPool&&)                 = delete; // No move constructor.
    MaxPool& operator=(const MaxPool&) = delete; // No copy assignment.
    MaxPool& operator=(MaxPool&&)      = delete; // No move assignment.

private:
    std::size_t poolSize() const noexcept;
    void checkDimensions(std::size_t inputSize, std::size_t poolSize) const noexcept;

    /** Pool input. */
    Matrix2d myInput;

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Pool output. */
    Matrix2d myOutput;
};
} // namespace ml::conv_layer
