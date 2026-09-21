/**
 * @file Neural network interface.
 */
#pragma once

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
     * @brief Predict based on the given input.
     *
     * @param[in] input Input to base the prediction on.
     *
     * @return Predicted output.
     */
    [[nodiscard]] virtual const Matrix1d& predict(const Matrix1d& input) noexcept = 0;
};
} // namespace ml::neural_network
