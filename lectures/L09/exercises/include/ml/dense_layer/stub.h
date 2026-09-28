/**
 * @file Dense layer stub.
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/dense_layer/interface.h"
#include "ml/types.h"

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
     * @brief Constructor.
     *
     * @param[in] nodeCount Number of nodes in the layer. Must be greater than 0.
     * @param[in] weightCount Weights per node. Must be greater than 0.
     * @param[in] outputValue Value every output element is set to (default = 0.5).
     */
    explicit Stub(const std::size_t nodeCount, const std::size_t weightCount,
                  const double outputValue = 0.5) noexcept
        : myWeights{}
        , myOutput{}
        , myError{}
        , myFeedforwardCount{}
    {
        if (0U == nodeCount)
        {
            std::fprintf(stderr, "Node count cannot be 0!\n");
            std::terminate();
        }
        if (0U == weightCount)
        {
            std::fprintf(stderr, "Weight count cannot be 0!\n");
            std::terminate();
        }
        myOutput.resize(nodeCount, outputValue);
        myError.resize(nodeCount);
        myWeights.resize(nodeCount, Matrix1d(weightCount));
    }

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Get dense layer outputs.
     *
     * @return Matrix holding the dense layer outputs.
     */
    [[nodiscard]] const Matrix1d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get computed error values for backpropagation.
     *
     * @return Matrix holding the computed error values.
     */
    [[nodiscard]] const Matrix1d& error() const noexcept override { return myError; }

    /**
     * @brief Get layer weights for backpropagation.
     *
     * @return Matrix holding the dense layer weights.
     */
    [[nodiscard]] const Matrix2d& weights() const noexcept override { return myWeights; }

    /**
     * @brief Get the node count.
     *
     * @return Number of nodes in the layer.
     */
    [[nodiscard]] std::size_t nodeCount() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the weight count of the layer.
     *
     * @return Number of weights per node in the layer.
     */
    [[nodiscard]] std::size_t weightCount() const noexcept override { return myWeights[0U].size(); }

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     */
    bool feedforward(const Matrix1d& input) noexcept override
    {
        ++myFeedforwardCount;
        return input.size() == weightCount();
    }

    /**
     * @brief Perform backpropagation in output layer.
     *
     * @param[in] reference Reference values. Must match the node count of this layer.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for output layers.
     */
    bool backpropagate(const Matrix1d& reference) noexcept override
    {
        return reference.size() == nodeCount();
    }

    /**
     * @brief Perform backpropagation in hidden layer.
     *
     * @param[in] nextLayer Next layer of the network.
     *
     * @return True on success, false on dimension mismatch.
     *
     * @note This method should only be used for hidden layers.
     */
    bool backpropagate(const Interface& nextLayer) noexcept override
    {
        return nextLayer.weightCount() == nodeCount();
    }

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Layer inputs. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on dimension mismatch or invalid learning rate.
     */
    bool optimize(const Matrix1d& input, const double learningRate) noexcept override
    {
        const bool lrValid{(0.0 < learningRate) && (1.0 > learningRate)};
        return lrValid && (input.size() == weightCount());
    }

    /**
     * @brief Set output values.
     *
     * @param[in] outputValue Value every output element is set to.
     */
    void setOutput(const double outputValue) noexcept
    {
        for (auto& value : myOutput)
        {
            value = outputValue;
        }
    }

    /**
     * @brief Get the number of times feedforward() has been called.
     *
     * @return Number of feedforward() calls during this run.
     */
    [[nodiscard]] std::size_t feedforwardCount() const noexcept { return myFeedforwardCount; }

    /**
     * @brief Clear the feedforward() call count.
     */
    void clearFeedforwardCount() noexcept { myFeedforwardCount = 0U; }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Layer weights. */
    Matrix2d myWeights;

    /** Layer output. */
    Matrix1d myOutput;

    /** Computed error values. */
    Matrix1d myError;

    /** Number of times feedforward() has been called. */
    std::size_t myFeedforwardCount;
};
} // namespace ml::dense_layer
