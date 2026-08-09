/**
 * @file Machine learning factory implementation details.
 */
#include <cstddef>
#include <memory>

#include "ml/act_func/none.h"
#include "ml/act_func/relu.h"
#include "ml/act_func/tanh.h"
#include "ml/conv_layer/conv.h"
#include "ml/conv_layer/max_pool.h"
#include "ml/dense_layer/dense.h"
#include "ml/factory/factory.h"
#include "ml/factory/stub.h"
#include "ml/flatten_layer/flatten.h"

namespace ml::factory
{
// -----------------------------------------------------------------------------
ActFuncPtr Factory::actFunc(const act_func::Type type) noexcept
{
    switch (type)
    {
        case act_func::Type::Relu:
            return std::make_unique<act_func::Relu>();
        case act_func::Type::Tanh:
            return std::make_unique<act_func::Tanh>();
        default:
            return std::make_unique<act_func::None>();
    }
}

// -----------------------------------------------------------------------------
ConvLayerPtr Factory::convLayer(const std::size_t inputSize, const std::size_t kernelSize,
                                const act_func::Type actFunc) noexcept
{
    ///! @todo Replace conv_layer::ConvStub with conv_layer::Conv when implemented.
    return std::make_unique<conv_layer::ConvStub>(inputSize, kernelSize, actFunc);
}

// -----------------------------------------------------------------------------
DenseLayerPtr Factory::denseLayer(const std::size_t inputSize, const std::size_t outputSize,
                                  const act_func::Type actFunc) noexcept
{
    return std::make_unique<dense_layer::Dense>(inputSize, outputSize, actFunc);
}

// -----------------------------------------------------------------------------
FlattenLayerPtr Factory::flattenLayer(const std::size_t inputSize) noexcept
{
    ///! @todo Replace flatten_layer::Stub with flatten_layer::Flatten when implemented.
    return std::make_unique<flatten_layer::Stub>(inputSize);
}

// -----------------------------------------------------------------------------
ConvLayerPtr Factory::maxPoolLayer(const std::size_t inputSize, const std::size_t poolSize) noexcept
{
    ///! @todo Replace conv_layer::MaxPoolStub with conv_layer::MaxPool when implemented.
    return std::make_unique<conv_layer::MaxPoolStub>(inputSize, poolSize);
}

// -----------------------------------------------------------------------------
FactoryPtr create(const bool stub) noexcept
{
    if (stub) { return std::make_unique<Stub>(); }
    return std::make_unique<Factory>();
}
} // namespace ml::factory
