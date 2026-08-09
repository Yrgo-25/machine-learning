/**
 * @file Dense layer stub.
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/act_func/type.h"
#include "ml/dense_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::dense_layer
{
/**
 * @brief Dense layer stub.
 *
 *        This class is non-copyable and non-movable.
 */
class Stub final : public Interface
{
public:
    /**
     * @brief Create a new dense layer.
     *
     * @param[in] inputSize Input size.
     * @param[in] outputSize Output size.
     * @param[in] actFunc Activation function to use for this layer (default = ReLU).
     */
    explicit Stub(const std::size_t inputSize, const std::size_t outputSize,
                  const act_func::Type actFunc = act_func::Type::Relu) noexcept
        : myInputGradients{}
        , myBias{}
        , myWeights{}
        , myOutput{}
        , myError{}
    {
        if (0U == outputSize)
        {
            std::fprintf(stderr, "Node count cannot be 0!\n");
            std::terminate();
        }
        else if (0U == inputSize)
        {
            std::fprintf(stderr, "Weight count cannot be 0!\n");
            std::terminate();
        }

        initMatrix(myInputGradients, inputSize);
        initMatrix(myBias, outputSize);
        initMatrix(myWeights, outputSize, inputSize);
        initMatrix(myOutput, outputSize);
        initMatrix(myError, outputSize);

        (void)(actFunc);
    }

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     *
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override
    {
        return myWeights.empty() ? 0U : myWeights[0U].size();
    }

    /**
     * @brief Get the output size of the layer.
     *
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the output values of the layer.
     *
     * @return Matrix holding the output values of the layer.
     */
    const Matrix1d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get the input gradients of the layer.
     *
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix1d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix1d& input) noexcept override
    {
        return matchDimensions(inputSize(), input.size());
    }

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override
    {
        return matchDimensions(outputSize(), outputGradients.size());
    }

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Matrix holding input data.
     * @param[in] learningRate Learning rate to use.
     *
     * @return True on success, false on failure.
     */
    bool optimize(const Matrix1d& input, const double learningRate) noexcept override
    {
        return matchDimensions(inputSize(), input.size()) && checkLearningRate(learningRate);
    }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Input gradients. */
    Matrix1d myInputGradients;

    /** Bias values. */
    Matrix1d myBias;

    /** Weights for each node. */
    Matrix2d myWeights;

    /** Output matrix. */
    Matrix1d myOutput;

    /** Error values. */
    Matrix1d myError;
};
} // namespace ml::dense_layer
