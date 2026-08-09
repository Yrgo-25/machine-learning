/**
 * @file Convolutional neural network (CNN) implementation details.
 */
#include <algorithm>
#include <cstddef>

#include "ml/cnn/cnn.h"
#include "ml/factory/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::cnn
{
// -----------------------------------------------------------------------------
Cnn::Cnn(factory::Interface& factory, const std::size_t convInput, const std::size_t convKernel,
         const act_func::Type convFunc, const std::size_t poolSize, const std::size_t denseOutput,
         const act_func::Type denseFunc) noexcept
    : myConvLayers{}
    , myDenseLayers{}
    , myFlattenLayer{nullptr}
    , myFactory{factory}
{
    myConvLayers.emplace_back(factory.convLayer(convInput, convKernel, convFunc));
    myConvLayers.emplace_back(factory.maxPoolLayer(convOutputSize(), poolSize));
    myFlattenLayer = factory.flattenLayer(convOutputSize());

    const std::size_t denseInput{myFlattenLayer->outputSize()};
    myDenseLayers.emplace_back(factory.denseLayer(denseInput, denseOutput, denseFunc));
}

// -----------------------------------------------------------------------------
Cnn::~Cnn() noexcept = default;

// -----------------------------------------------------------------------------
std::size_t Cnn::inputSize() const noexcept { return myConvLayers[0U]->inputSize(); }

// -----------------------------------------------------------------------------
std::size_t Cnn::outputSize() const noexcept
{
    const std::size_t last{myDenseLayers.size() - 1U};
    return myDenseLayers[last]->outputSize();
}

// -----------------------------------------------------------------------------
const Matrix1d& Cnn::predict(const Matrix2d& input) noexcept
{
    feedforward(input);
    return output();
}

// -----------------------------------------------------------------------------
void Cnn::addDenseLayer(const std::size_t outputSize, const act_func::Type actFunc) noexcept
{
    myDenseLayers.emplace_back(myFactory.denseLayer(this->outputSize(), outputSize, actFunc));
}

// -----------------------------------------------------------------------------
bool Cnn::train(const Matrix3d& trainIn, const Matrix2d& trainOut, const std::size_t epochCount,
                const double learningRate) noexcept
{
    if ((0.0 >= learningRate) || (1.0 < learningRate)) { return false; }
    else if (0U == epochCount) { return false; }

    const auto setCount = std::min(trainIn.size(), trainOut.size());

    if (0U == setCount) { return false; }

    TrainOrderList trainOrder{createTrainOrderList(setCount)};

    for (std::size_t i{}; i < epochCount; ++i)
    {
        shuffleTrainOrderList(trainOrder);

        for (auto& j : trainOrder)
        {
            const Matrix2d& input{trainIn[j]};
            const Matrix1d& output{trainOut[j]};

            const bool success{feedforward(input) && backpropagate(output) &&
                               optimize(learningRate)};
            if (!success) { return false; }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
const Matrix1d& Cnn::output() const noexcept
{
    const std::size_t last{myDenseLayers.size() - 1U};
    return myDenseLayers[last]->output();
}

// -----------------------------------------------------------------------------
const Matrix2d& Cnn::convOutput() const noexcept
{
    const std::size_t last{myConvLayers.size() - 1U};
    return myConvLayers[last]->output();
}

// -----------------------------------------------------------------------------
std::size_t Cnn::convOutputSize() const noexcept
{
    const std::size_t last{myConvLayers.size() - 1U};
    return myConvLayers[last]->outputSize();
}

// -----------------------------------------------------------------------------
bool Cnn::feedforward(const Matrix2d& input) noexcept
{
    bool success{myConvLayers[0U]->feedforward(input)};

    for (std::size_t i{1U}; i < myConvLayers.size(); ++i)
    {
        auto& prevLayer{*(myConvLayers[i - 1U])};
        success &= myConvLayers[i]->feedforward(prevLayer.output());
    }
    if (!success) { return false; }

    success = myFlattenLayer->feedforward(convOutput());
    if (!success) { return false; }

    success = myDenseLayers[0U]->feedforward(myFlattenLayer->output());

    for (std::size_t i{1U}; i < myDenseLayers.size(); ++i)
    {
        auto& prevLayer{*(myDenseLayers[i - 1U])};
        success &= myDenseLayers[i]->feedforward(prevLayer.output());
    }
    return success;
}

// -----------------------------------------------------------------------------
bool Cnn::backpropagate(const Matrix1d& output) noexcept
{
    const std::size_t lastDense{myDenseLayers.size() - 1U};
    bool success{myDenseLayers[lastDense]->backpropagate(output)};

    for (std::size_t i{lastDense}; i > 0U; --i)
    {
        success &= myDenseLayers[i - 1U]->backpropagate(myDenseLayers[i]->inputGradients());
    }
    if (!success) { return false; }

    success = myFlattenLayer->backpropagate(myDenseLayers[0U]->inputGradients());
    if (!success) { return false; }

    const std::size_t lastConv{myConvLayers.size() - 1U};
    success = myConvLayers[lastConv]->backpropagate(myFlattenLayer->inputGradients());

    for (std::size_t i{lastConv}; i > 0U; --i)
    {
        success &= myConvLayers[i - 1U]->backpropagate(myConvLayers[i]->inputGradients());
    }
    return success;
}

// -----------------------------------------------------------------------------
bool Cnn::optimize(const double learningRate) noexcept
{
    bool success{myConvLayers[0U]->optimize(learningRate)};

    for (std::size_t i{1U}; i < myConvLayers.size(); ++i)
    {
        success &= myConvLayers[i]->optimize(learningRate);
    }
    if (!success) { return false; }

    success = myDenseLayers[0U]->optimize(myFlattenLayer->output(), learningRate);

    for (std::size_t i{1U}; i < myDenseLayers.size(); ++i)
    {
        auto& prevLayer{*(myDenseLayers[i - 1U])};
        success &= myDenseLayers[i]->optimize(prevLayer.output(), learningRate);
    }
    return success;
}
} // namespace ml::cnn
