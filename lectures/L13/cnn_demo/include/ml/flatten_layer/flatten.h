/**
 * @file Flatten layer implementation.
 */
#pragma once

#include "ml/flatten_layer/interface.h"
#include "ml/types.h"

namespace ml::flatten_layer
{
/**
 * @brief Flatten layer implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Flatten final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     */
    explicit Flatten(std::size_t inputSize) noexcept;

    /**
     * @brief Destructor.
     */
    ~Flatten() noexcept override = default;

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
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override;

    /**
     * @brief Get the flattened output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    const Matrix1d& output() const noexcept override;

    /**
     * @brief Flatten the input from 2D to 1D.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override;

    /**
     * @brief Unflatten the output gradients from 1D to 2D.
     *
     * @param[in] outputGradients Matrix holding output gradients.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override;

    Flatten()                          = delete; // No default constructor.
    Flatten(const Flatten&)            = delete; // No copy constructor.
    Flatten(Flatten&&)                 = delete; // No move constructor.
    Flatten& operator=(const Flatten&) = delete; // No copy assignment.
    Flatten& operator=(Flatten&&)      = delete; // No move assignment.

private:
    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Flattened output. */
    Matrix1d myOutput;
};
} // namespace ml::flatten_layer
