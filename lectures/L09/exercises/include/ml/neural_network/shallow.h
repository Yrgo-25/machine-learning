/**
 * @file Shallow neural network implementation.
 */
#pragma once

#include <cstddef>

#include "ml/neural_network/interface.h"
#include "ml/types.h"

// clang-format off

/** Dense layer interface. */
namespace ml::dense_layer { class Interface; }
// clang-format on

namespace ml::neural_network
{
/**
 * @brief Shallow neural network (only one dense layer) implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Shallow final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] hiddenLayer The network's single hidden layer.
     * @param[in] outputLayer The network's output layer.
     * @param[in] trainInput Training input sets. Size must be greater than 0.
     * @param[in] trainOutput Training output sets. Size must be greater than 0.
     *
     * @note The layers are fully connected: for this reason, the node count of the hidden layer
     *       must match the weight count of the output layer.
     *       Also, at least one full training set must be provided.
     */
    explicit Shallow(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                     const Matrix2d& trainInput, const Matrix2d& trainOutput) noexcept;

    /**
     * @brief Destructor.
     */
    ~Shallow() noexcept override = default;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input to base the prediction on.
     *
     * @return Predicted output.
     */
    [[nodiscard]] const Matrix1d& predict(const Matrix1d& input) noexcept override;

    /**
     * @brief Train the network.
     *
     * @param[in] epochCount Number of epochs to perform training. Must be greater than 0.
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on invalid parameters.
     */
    bool train(std::size_t epochCount, double learningRate = 0.01) noexcept;

    Shallow()                          = delete; // No default constructor.
    Shallow(const Shallow&)            = delete; // No copy constructor.
    Shallow(Shallow&&)                 = delete; // No move constructor.
    Shallow& operator=(const Shallow&) = delete; // No copy assignment.
    Shallow& operator=(Shallow&&)      = delete; // No move assignment.

private:
    bool feedforward(const Matrix1d& input) noexcept;
    bool backpropagate(const Matrix1d& output) noexcept;
    bool optimize(const Matrix1d& input, double learningRate) noexcept;

    /** Hidden layer. */
    ml::dense_layer::Interface& myHiddenLayer;

    /** Output layer. */
    ml::dense_layer::Interface& myOutputLayer;

    /** Training input. */
    const Matrix2d& myTrainInput;

    /** Training output. */
    const Matrix2d& myTrainOutput;

    /** Number of training sets. */
    const std::size_t myTrainSetCount;
};
} // namespace ml::neural_network
