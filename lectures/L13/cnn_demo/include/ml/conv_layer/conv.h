/**
 * @file Convolutional layer implementation.
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/act_func/type.h"
#include "ml/conv_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
/**
 * @brief Convolutional layer implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Conv final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use (default = none).
     */
    explicit Conv(std::size_t inputSize, std::size_t kernelSize,
                  act_func::Type actFunc = act_func::Type::None) noexcept;

    /**
     * @brief Destructor.
     */
    ~Conv() noexcept override;

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
    const Matrix2d& inputGradients() const noexcept;

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
     * @brief Perform optimization.
     *
     * @param[in] learningRate Learning rate to use.
     *
     * @return True on success, false on failure.
     */
    bool optimize(double learningRate) noexcept override;

    Conv()                       = delete; // No default constructor.
    Conv(const Conv&)            = delete; // No copy constructor.
    Conv(Conv&&)                 = delete; // No move constructor.
    Conv& operator=(const Conv&) = delete; // No copy assignment.
    Conv& operator=(Conv&&)      = delete; // No move assignment.

private:
    void checkParameters(std::size_t inputSize, std::size_t kernelSize) noexcept;
    void init(std::size_t inputSize, std::size_t kernelSize, act_func::Type actFunc) noexcept;
    bool padInput(const Matrix2d& input) noexcept;
    void extractUnpaddedGradients() noexcept;

    /** Minimum valid kernel size. */
    static constexpr std::size_t MinKernelSize{1U};

    /** Minimum valid kernel size. */
    static constexpr std::size_t MaxKernelSize{11U};

    /** Input matrix (padded with zeros). */
    Matrix2d myInputPadded;

    /** Input gradients (padded with zeros). */
    Matrix2d myInputGradientsPadded;

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Kernel matrix. */
    Matrix2d myKernel;

    /** Kernel gradients. */
    Matrix2d myKernelGradients;

    /** Output matrix. */
    Matrix2d myOutput;

    /** Pre-activation output matrix. */
    Matrix2d myPreActivationOutput;

    /** Bias value. */
    double myBias;

    /** Bias gradient. */
    double myBiasGradient;

    /** Activation function. */
    ActFuncPtr myActFunc;
};
} // namespace ml::conv_layer
