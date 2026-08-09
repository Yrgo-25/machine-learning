/**
 * @file Convolutional neural network (CNN) interface.
 */
#pragma once

#include <cstddef>

#include "ml/types.h"

namespace ml::cnn
{
/**
 * @brief Convolutional neural network (CNN) interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Get the input size of the CNN.
     *
     * @return The input size of the CNN.
     */
    [[nodiscard]] virtual std::size_t inputSize() const noexcept = 0;

    /**
     * @brief Get the output size of the CNN.
     *
     * @return The output size of the CNN.
     */
    [[nodiscard]] virtual std::size_t outputSize() const noexcept = 0;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input for which to predict.
     *
     * @return The predicted output.
     */
    [[nodiscard]] virtual const Matrix1d& predict(const Matrix2d& input) noexcept = 0;
};
} // namespace ml::cnn