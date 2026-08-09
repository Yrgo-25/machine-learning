/**
 * @file Convolutional layer stubs.
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
 * @brief Convolutional layer stub.
 *
 *        This class is non-copyable and non-movable.
 */
class ConvStub final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use (default = none).
     */
    explicit ConvStub(const std::size_t inputSize, const std::size_t kernelSize,
                      const act_func::Type actFunc = act_func::Type::None) noexcept
        : myInputGradients{}
        , myKernel{}
        , myOutput{}
    {
        if ((MinKernelSize > kernelSize) || (MaxKernelSize < kernelSize))
        {
            std::fprintf(stderr,
                         "Invalid kernel size %zu: kernel size must be in range [%zu, %zu]!\n",
                         kernelSize, MinKernelSize, MaxKernelSize);
            std::terminate();
        }
        else if (inputSize < kernelSize)
        {
            std::fprintf(stderr, "Failed to create convolutional layer: kernel size cannot be "
                                 "greater than input size!\n");
            std::terminate();
        }

        initMatrix(myInputGradients, inputSize);
        initMatrix(myKernel, kernelSize);
        initMatrix(myOutput, inputSize);

        (void)(actFunc);
    }

    /**
     * @brief Destructor.
     */
    ~ConvStub() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     *
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override { return myInputGradients.size(); }

    /**
     * @brief Get the output size of the layer.
     *
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override
    {
        return matchDimensions(myOutput.size(), input.size()) && isMatrixSquare(input);
    }

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override
    {
        return matchDimensions(myOutput.size(), outputGradients.size()) &&
               isMatrixSquare(outputGradients);
    }

    /**
     * @brief Perform optimization.
     *
     * @param[in] learningRate Learning rate to use.
     *
     * @return True on success, false on failure.
     */
    bool optimize(const double learningRate) noexcept override
    {
        return checkLearningRate(learningRate);
    }

    ConvStub()                           = delete; // No default constructor.
    ConvStub(const ConvStub&)            = delete; // No copy constructor.
    ConvStub(ConvStub&&)                 = delete; // No move constructor.
    ConvStub& operator=(const ConvStub&) = delete; // No copy assignment.
    ConvStub& operator=(ConvStub&&)      = delete; // No move assignment.

private:
    /** Minimum valid kernel size. */
    static constexpr std::size_t MinKernelSize{1U};

    /** Minimum valid kernel size. */
    static constexpr std::size_t MaxKernelSize{11U};

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Kernel matrix. */
    Matrix2d myKernel;

    /** Output matrix. */
    Matrix2d myOutput;
};

/**
 * @brief Max pooling layer stub.
 */
class MaxPoolStub final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     */
    explicit MaxPoolStub(const std::size_t inputSize, const std::size_t poolSize) noexcept
        : myInput{}
        , myInputGradients{}
        , myOutput{}
    {
        if (0U == inputSize)
        {
            std::fprintf(stderr, "Input size cannot be 0!\n");
            std::terminate();
        }
        else if (0U == poolSize)
        {
            std::fprintf(stderr, "Pool size cannot be 0!\n");
            std::terminate();
        }
        else if (inputSize < poolSize)
        {
            std::fprintf(stderr, "Input size cannot be smaller than the pool size!\n");
            std::terminate();
        }
        else if (0 != (inputSize % poolSize))
        {
            std::fprintf(stderr, "Input size must be divisible by pool size!\n");
            std::terminate();
        }

        const std::size_t outputSize{inputSize / poolSize};
        initMatrix(myInput, inputSize);
        initMatrix(myInputGradients, inputSize);
        initMatrix(myOutput, outputSize);
    }

    /**
     * @brief Default destructor.
     */
    ~MaxPoolStub() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     *
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override { return myInputGradients.size(); }

    /**
     * @brief Get the output size of the layer.
     *
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the output of the layer.
     *
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override
    {
        return matchDimensions(myInput.size(), input.size()) && isMatrixSquare(input);
    }

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override
    {
        return matchDimensions(myOutput.size(), outputGradients.size()) &&
               isMatrixSquare(outputGradients);
    }

    /**
     * @brief Perform optimization (not implemented for pooling layers).
     *
     * @param[in] learningRate Learning rate to use.
     *
     * @return True (optimization is a no-op for pooling layers).
     */
    bool optimize(const double learningRate) noexcept override
    {
        (void)(learningRate);
        return true;
    }

    MaxPoolStub()                              = delete; // No default constructor.
    MaxPoolStub(const MaxPoolStub&)            = delete; // No copy constructor.
    MaxPoolStub(MaxPoolStub&&)                 = delete; // No move constructor.
    MaxPoolStub& operator=(const MaxPoolStub&) = delete; // No copy assignment.
    MaxPoolStub& operator=(MaxPoolStub&&)      = delete; // No move assignment.

private:
    /** Pool input. */
    Matrix2d myInput;

    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Pool output. */
    Matrix2d myOutput;
};
} // namespace ml::conv_layer
