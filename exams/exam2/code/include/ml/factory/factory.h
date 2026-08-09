/**
 * @file Machine learning factory.
 */
#pragma once

#include <cstddef>

#include "ml/factory/interface.h"

namespace ml::factory
{
/**
 * @brief Machine learning factory.
 *
 *        This class is non-copyable and non-movable.
 */
class Factory final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    Factory() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Factory() noexcept override = default;

    /**
     * @brief Create an activation function.
     *
     * @param[in] type The type of activation function to create.
     *
     * @return Pointer to the new activation function.
     */
    ActFuncPtr actFunc(act_func::Type type) noexcept override;

    /**
     * @brief Create a convolutional layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new convolutional layer.
     */
    ConvLayerPtr convLayer(std::size_t inputSize, std::size_t kernelSize,
                           act_func::Type actFunc) noexcept override;

    /**
     * @brief Create a dense layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] outputSize Output size. Must be greater than 0.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new dense layer.
     */
    DenseLayerPtr denseLayer(std::size_t inputSize, std::size_t outputSize,
                             act_func::Type actFunc) noexcept override;

    /**
     * @brief Create a flatten layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     *
     * @return Pointer to the new flatten layer.
     */
    FlattenLayerPtr flattenLayer(std::size_t inputSize) noexcept override;

    /**
     * @brief Create a max pooling layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     *
     * @return Pointer to the new max pooling layer.
     */
    ConvLayerPtr maxPoolLayer(std::size_t inputSize, std::size_t poolSize) noexcept override;

    Factory(const Factory&)            = delete; // No copy constructor.
    Factory(Factory&&)                 = delete; // No move constructor.
    Factory& operator=(const Factory&) = delete; // No copy assignment.
    Factory& operator=(Factory&&)      = delete; // No move assignment.
};

/**
 * @brief Create a factory.
 *
 * @param[in] stub True to create a stub factory (default = false).
 *
 * @return Pointer to the new factory.
 */
[[nodiscard]] FactoryPtr create(bool stub = false) noexcept;

} // namespace ml::factory