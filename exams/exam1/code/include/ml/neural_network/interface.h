/**
 * @file Neural network interface.
 */
#pragma once

#include <cstdint>

#include "ml/types.h"

namespace ml::neural_network
{
/**
 * @brief Neural network interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Get the input size of the network.
     *
     * @return Number of nodes in the input layer.
     */
    [[nodiscard]] virtual std::uint16_t inputSize() const noexcept = 0;

    /**
     * @brief Get the output size of the network.
     *
     * @return Number of nodes in the output layer.
     */
    [[nodiscard]] virtual std::uint16_t outputSize() const noexcept = 0;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input values.
     *
     * @return Predicted output.
     */
    [[nodiscard]] virtual const Matrix1d& predict(const Matrix1d& input) noexcept = 0;
};
} // namespace ml::neural_network
