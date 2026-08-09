/**
 * @file Machine learning factory interface.
 */
#pragma once

#include <cstddef>

#include "ml/act_func/interface.h"
#include "ml/act_func/type.h"
#include "ml/conv_layer/interface.h"
#include "ml/dense_layer/interface.h"
#include "ml/flatten_layer/interface.h"
#include "ml/types.h"

namespace ml::factory
{
/**
 * @brief Machine learning factory interface.
 */
class Interface
{
public:
    /**
     * @brief Destructor.
     */
    virtual ~Interface() noexcept = default;

    /**
     * @brief Create an activation function.
     *
     * @param[in] type The type of activation function to create.
     *
     * @return Pointer to the new activation function.
     */
    [[nodiscard]] virtual ActFuncPtr actFunc(act_func::Type type) noexcept = 0;

    /**
     * @brief Create a convolutional layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new convolutional layer.
     */
    [[nodiscard]] virtual ConvLayerPtr convLayer(std::size_t inputSize, std::size_t kernelSize,
                                                 act_func::Type actFunc) noexcept = 0;

    /**
     * @brief Create a dense layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] outputSize Output size. Must be greater than 0.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new dense layer.
     */
    [[nodiscard]] virtual DenseLayerPtr denseLayer(std::size_t inputSize, std::size_t outputSize,
                                                   act_func::Type actFunc) noexcept = 0;

    /**
     * @brief Create a flatten layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     *
     * @return Pointer to the new flatten layer.
     */
    [[nodiscard]] virtual FlattenLayerPtr flattenLayer(std::size_t inputSize) noexcept = 0;

    /**
     * @brief Create a max pooling layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     *
     * @return Pointer to the new max pooling layer.
     */
    [[nodiscard]] virtual ConvLayerPtr maxPoolLayer(std::size_t inputSize,
                                                    std::size_t poolSize) noexcept = 0;
};
} // namespace ml::factory