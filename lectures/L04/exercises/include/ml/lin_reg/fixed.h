/**
 * @file Linear regression model with fixed learning rate.
 */
#pragma once

#include <cstddef>

#include "ml/lin_reg/interface.h"
#include "ml/types.h"

namespace ml::lin_reg
{
/**
 * @brief Linear regression model with fixed learning rate.
 *
 *        This class is non-copyable and non-movable.
 */
class Fixed final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] trainIn Training inputs. Size must be greater than 0.
     * @param[in] trainOut Training outputs. Size must be greater than 0.
     *
     * @note The user is responsible for the lifetime of the training data; the training data
     *       must be accessible during the training cycle. The training data must not change
     *       during training, else the performance might suffer.
     */
    explicit Fixed(const Matrix1d& trainIn, const Matrix1d& trainOut) noexcept;

    /**
     * @brief Destructor.
     */
    ~Fixed() noexcept override = default;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input to predict with.
     *
     * @return Predicted output.
     */
    [[nodiscard]] double predict(double input) const noexcept override;

    /**
     * @brief Train the model for the given number of epochs.
     *
     * @param[in] epochCount Number of epochs to train the model. Must be greater than 0.
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0), non-inclusive.
     * @param[in] precisionThreshold Precision threshold. Training is aborted if/when this
     *                               threshold is reached.
     *
     * @return True if training was performed, false if input argument is invalid.
     */
    bool train(std::size_t epochCount, double learningRate = 0.01,
               double precisionThreshold = 0.999999) noexcept;

    Fixed()                        = delete; // No default constructor.
    Fixed(const Fixed&)            = delete; // No copy constructor.
    Fixed(Fixed&&)                 = delete; // No move constructor.
    Fixed& operator=(const Fixed&) = delete; // No copy assignment.
    Fixed& operator=(Fixed&&)      = delete; // No move assignment.

private:
    /**
     * @brief Adjust the trainable parameters.
     *
     * @param[in] input Input value (x).
     * @param[in] output Output value (yref).
     * @param[in] learningRate Learning rate.
     */
    void optimize(double input, double output, double learningRate) noexcept;

    /**
     * @brief Shuffle training order index list.
     */
    void shuffle() noexcept;

    /**
     * @brief Compute precision.
     *
     * @return Precision as a floating-point value.
     */
    [[nodiscard]] double computePrecision() const noexcept;

    /** Matrix holding the training order via index. */
    MatrixU32 myTrainOrder;

    /** Matrix holding training inputs. */
    const Matrix1d& myTrainIn;

    /** Matrix holding training outputs. */
    const Matrix1d& myTrainOut;

    /** Bias value (m). */
    double myBias;

    /** Weight value (k). */
    double myWeight;
};
} // namespace ml::lin_reg
