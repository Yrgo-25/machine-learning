/**
 * @file Dense layer stub.
 */
#pragma once

#include <cstdint>
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
     * @param[in] weightCount Number of weights per node in the layer. Must be greater than 0.
     * @param[in] actFunc Activation function (default = ReLU).
     */
    explicit Stub(const std::uint16_t nodeCount, const std::uint16_t weightCount,
                  const ActFunc actFunc = ActFunc::Relu) noexcept
        : myWeights(nodeCount, Matrix1d(weightCount))
        , myOutput(nodeCount)
        , myError(nodeCount)
    {
        (void)(actFunc);

        if ((0U == nodeCount) || (0U == weightCount))
        {
            std::fprintf(stderr, "Failed to create dense layer: "
                                 "node and weight count must both be greater than 0!\n");
            std::terminate();
        }
        myWeights.resize(nodeCount, Matrix1d(weightCount));
        myOutput.resize(nodeCount);
        myError.resize(nodeCount);
    }

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Get layer output.
     *
     * @return Layer output.
     */
    const Matrix1d& output() const noexcept override { return myOutput; }

    /**
     * @brief Get computed error values.
     *
     * @return Computed error values for each node in the layer.
     */
    const Matrix1d& error() const noexcept override { return myError; }

    /**
     * @brief Get layer weights.
     *
     * @return Weights for each node in the layer.
     */
    const Matrix2d& weights() const noexcept override { return myWeights; }

    /**
     * @brief Get the node count of this layer.
     *
     * @return Number of nodes in this layer.
     */
    std::uint16_t nodeCount() const noexcept override
    {
        return static_cast<std::uint16_t>(myOutput.size());
    }

    /**
     * @brief Get the weight count of this layer.
     *
     * @return Number of weights per node in this layer.
     */
    std::uint16_t weightCount() const noexcept override
    {
        const auto count = !myWeights.empty() ? myWeights[0U].size() : 0U;
        return static_cast<std::uint16_t>(count);
    }

    /**
     * @brief Perform feedforward.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     *
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix1d& input) noexcept override
    {
        const auto size     = static_cast<std::uint16_t>(input.size());
        const auto mismatch = size != weightCount();

        if (mismatch)
        {
            std::fprintf(stderr,
                         "Feedforward failed due to dimension mismatch: "
                         "expected %u, actual = %u!\n",
                         weightCount(), size);
            return false;
        }
        return true;
    }

    /**
     * @brief Perform backpropagation for output layer.
     *
     * @param[in] output Output values. Must match the node count of this layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& output) noexcept override
    {
        const auto size     = static_cast<std::uint16_t>(output.size());
        const auto mismatch = size != nodeCount();

        if (mismatch)
        {
            std::fprintf(stderr,
                         "Backpropagation failed due to dimension mismatch: "
                         "expected %u, actual = %u!\n",
                         nodeCount(), size);
            return false;
        }
        return true;
    }

    /**
     * @brief Perform backpropagation for hidden layer.
     *
     * @param[in] nextLayer Next layer. Must match the dimensions of this layer.
     *
     * @return True on success, false on failure.
     */
    bool backpropagate(const Interface& nextLayer) noexcept override
    {
        const auto mismatch = nextLayer.weightCount() != nodeCount();

        if (mismatch)
        {
            std::fprintf(stderr,
                         "Backpropagation failed due to dimension mismatch: "
                         "expected %u, actual = %u!\n",
                         nodeCount(), nextLayer.weightCount());
            return false;
        }
        return true;
    }

    /**
     * @brief Perform optimization.
     *
     * @param[in] input Input values. Must match the weight count of this layer.
     * @param[in] learningRate Learning rate. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on failure.
     */
    bool optimize(const Matrix1d& input, const double learningRate) noexcept override
    {
        const auto lrValid = ((0.0 < learningRate) && (1.0 > learningRate));

        if (!lrValid)
        {
            std::fprintf(stderr, "Optimization failed due to invalid learning rate %g!\n",
                         learningRate);
            return false;
        }

        const auto size     = static_cast<std::uint16_t>(input.size());
        const auto mismatch = size != weightCount();

        if (mismatch)
        {
            std::fprintf(stderr,
                         "Optimization failed due to dimension mismatch: "
                         "expected %u, actual = %u!\n",
                         weightCount(), size);
            return false;
        }
        return true;
    }

    /**
     * @brief Set dense layer output.
     *
     * @param[in] output Output values. Must match the node count of this layer.
     */
    void setOutput(const Matrix1d& output) noexcept
    {
        const auto size     = static_cast<std::uint16_t>(output.size());
        const auto mismatch = size != nodeCount();

        if (mismatch)
        {
            std::fprintf(stderr, "Output dimension mismatch: expected %u, actual = %u!\n",
                         nodeCount(), size);
            std::terminate();
        }
        myOutput = output;
    }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Layer weights. */
    Matrix2d myWeights;

    /** Output values. */
    Matrix1d myOutput;

    /** Computer error values. */
    Matrix1d myError;
};
} // namespace ml::dense_layer
