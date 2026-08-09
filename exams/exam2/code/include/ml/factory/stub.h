/**
 * @file Machine learning factory stub.
 */
#pragma once

#include <cstddef>
#include <memory>

#include "ml/act_func/none.h"
#include "ml/conv_layer/stub.h"
#include "ml/dense_layer/stub.h"
#include "ml/factory/interface.h"
#include "ml/flatten_layer/stub.h"

namespace ml::factory
{
/**
 * @brief Machine learning factory stub.
 *
 *        This class is non-copyable and non-movable.
 */
class Stub final : public Interface
{
public:
    /**
     * @brief Constructor.
     */
    Stub() noexcept = default;

    /**
     * @brief Destructor.
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Create an activation function.
     *
     * @param[in] type The type of activation function to create.
     *
     * @return Pointer to the new activation function.
     */
    ActFuncPtr actFunc(const act_func::Type type) noexcept override
    {
        (void)(type);
        return std::make_unique<act_func::None>();
    }

    /**
     * @brief Create a convolutional layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new convolutional layer.
     */
    ConvLayerPtr convLayer(const std::size_t inputSize, const std::size_t kernelSize,
                           const act_func::Type actFunc) noexcept override
    {
        return std::make_unique<conv_layer::ConvStub>(inputSize, kernelSize, actFunc);
    }

    /**
     * @brief Create a dense layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] outputSize Output size. Must be greater than 0.
     * @param[in] actFunc Activation function to use.
     *
     * @return Pointer to the new dense layer.
     */
    DenseLayerPtr denseLayer(const std::size_t inputSize, const std::size_t outputSize,
                             const act_func::Type actFunc) noexcept override
    {
        return std::make_unique<dense_layer::Stub>(inputSize, outputSize, actFunc);
    }

    /**
     * @brief Create a flatten layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     *
     * @return Pointer to the new flatten layer.
     */
    FlattenLayerPtr flattenLayer(const std::size_t inputSize) noexcept override
    {
        return std::make_unique<flatten_layer::Stub>(inputSize);
    }

    /**
     * @brief Create a max pooling layer.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] poolSize Pool size. Must divide the input size.
     *
     * @return Pointer to the new max pooling layer.
     */
    ConvLayerPtr maxPoolLayer(const std::size_t inputSize,
                              const std::size_t poolSize) noexcept override
    {
        return std::make_unique<conv_layer::MaxPoolStub>(inputSize, poolSize);
    }

    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.
};
} // namespace ml::factory