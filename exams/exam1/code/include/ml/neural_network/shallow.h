/**
 * @file Shallow neural network implementation.
 */
#pragma once

#include <cstdint>

#include "ml/neural_network/interface.h"
#include "ml/types.h"

namespace ml::neural_network
{
/**
 * @brief Shallow neural network implementation.
 *
 *        This class is non-copyable and non-movable.
 */
class Shallow final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] hiddenLayer Hidden layer.
     * @param[in] outputLayer Output layer.
     *
     * @note The weight count of the output layer must match the node count of the hidden layer.
     */
    explicit Shallow(IDenseLayer& hiddenLayer, IDenseLayer& outputLayer) noexcept;

    /**
     * @brief Destructor.
     */
    ~Shallow() noexcept override = default;

    /**
     * @brief Get the input size of the network.
     *
     * @return Number of nodes in the input layer.
     */
    std::uint16_t inputSize() const noexcept override;

    /**
     * @brief Get the output size of the network.
     *
     * @return Number of nodes in the output layer.
     */
    std::uint16_t outputSize() const noexcept override;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input values.
     *
     * @return Predicted output.
     */
    const Matrix1d& predict(const Matrix1d& input) noexcept override;

    /**
     * @brief Train the network for the given number of epochs.
     *
     * @param[in] trainIn Input data.
     * @param[in] trainOut Output data.
     * @param[in] epochCount Number of epochs to train the network. Must be greater than 0.
     * @param[in] learningRate Learning rate. Must be in range (0.0, 1.0).
     *
     * @return True on success, false on failure.
     */
    bool train(const Matrix2d& trainIn, const Matrix2d& trainOut, std::uint32_t epochCount,
               double learningRate) noexcept;

    Shallow()                          = delete; // No default constructor.
    Shallow(const Shallow&)            = delete; // No copy constructor.
    Shallow(Shallow&&)                 = delete; // No move constructor.
    Shallow& operator=(const Shallow&) = delete; // No copy assignment.
    Shallow& operator=(Shallow&&)      = delete; // No move assignment.

private:
    void feedforward(const Matrix1d& input) noexcept;
    void backpropagate(const Matrix1d& output) noexcept;
    void optimize(const Matrix1d& input, double learningRate) noexcept;

    /** Hidden layer. */
    IDenseLayer& myHiddenLayer;

    /** Output layer. */
    IDenseLayer& myOutputLayer;
};
} // namespace ml::neural_network
