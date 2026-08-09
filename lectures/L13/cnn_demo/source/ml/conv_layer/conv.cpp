/**
 * @file Convolutional layer implementation details.
 */
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/conv_layer/conv.h"
#include "ml/factory/factory.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
// -----------------------------------------------------------------------------
Conv::Conv(const std::size_t inputSize, const std::size_t kernelSize,
           const act_func::Type actFunc) noexcept
    : myInputPadded{}
    , myInputGradientsPadded{}
    , myInputGradients{}
    , myKernel{}
    , myKernelGradients{}
    , myOutput{}
    , myPreActivationOutput{}
    , myBias{randomStartVal()}
    , myBiasGradient{}
    , myActFunc{nullptr}
{
    checkParameters(inputSize, kernelSize);
    init(inputSize, kernelSize, actFunc);
}

// -----------------------------------------------------------------------------
Conv::~Conv() noexcept = default;

// -----------------------------------------------------------------------------
std::size_t Conv::inputSize() const noexcept { return myInputGradients.size(); }

// -----------------------------------------------------------------------------
std::size_t Conv::outputSize() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
const Matrix2d& Conv::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const Matrix2d& Conv::inputGradients() const noexcept { return myInputGradients; }

// -----------------------------------------------------------------------------
bool Conv::feedforward(const Matrix2d& input) noexcept
{
    if (!matchDimensions(myOutput.size(), input.size())) { return false; }
    if (!padInput(input)) { return false; }

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            auto sum = myBias;

            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    sum += myInputPadded[i + ki][j + kj] * myKernel[ki][kj];
                }
            }
            myPreActivationOutput[i][j] = sum;
            myOutput[i][j]              = myActFunc->output(sum);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Conv::backpropagate(const Matrix2d& outputGradients) noexcept
{
    if (!matchDimensions(myOutput.size(), outputGradients.size())) { return false; }
    initMatrix(myInputGradientsPadded);
    initMatrix(myInputGradients);
    initMatrix(myKernelGradients);
    myBiasGradient = 0.0;

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        if (!isMatrixSquare(outputGradients, i)) { return false; }

        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            const auto delta   = myActFunc->delta(myPreActivationOutput[i][j]);
            const auto outGrad = outputGradients[i][j] * delta;

            myBiasGradient += outGrad;

            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    myKernelGradients[ki][kj] += myInputPadded[i + ki][j + kj] * outGrad;
                    myInputGradientsPadded[i + ki][j + kj] += myKernel[ki][kj] * outGrad;
                }
            }
        }
    }
    extractUnpaddedGradients();
    return true;
}

// -----------------------------------------------------------------------------
bool Conv::optimize(const double learningRate) noexcept
{
    if (!checkLearningRate(learningRate)) { return false; }
    myBias += myBiasGradient * learningRate;

    for (std::size_t i{}; i < myKernel.size(); ++i)
    {
        for (std::size_t j{}; j < myKernel.size(); ++j)
        {
            myKernel[i][j] += myKernelGradients[i][j] * learningRate;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Conv::checkParameters(const std::size_t inputSize, const std::size_t kernelSize) noexcept
{
    if ((MinKernelSize > kernelSize) || (MaxKernelSize < kernelSize))
    {
        std::fprintf(stderr, "Invalid kernel size %zu: kernel size must be in range [%zu,%zu]!\n",
                     kernelSize, MinKernelSize, MaxKernelSize);
        std::terminate();
    }
    if (inputSize < kernelSize)
    {
        std::fprintf(stderr, "Failed to create convolutional layer: "
                             "kernel size cannot be greater than input size!\n");
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
void Conv::init(const std::size_t inputSize, const std::size_t kernelSize,
                const act_func::Type actFunc) noexcept
{
    const auto offset     = kernelSize / 2U;
    const auto paddedSize = inputSize + (2U * offset);

    initMatrix(myInputPadded, paddedSize);
    initMatrix(myInputGradientsPadded, paddedSize);
    initMatrix(myInputGradients, inputSize);

    initMatrix(myKernel, kernelSize);
    initMatrix(myKernelGradients, kernelSize);

    initMatrix(myOutput, inputSize);
    initMatrix(myPreActivationOutput, inputSize);

    for (std::size_t i{}; i < kernelSize; ++i)
    {
        for (std::size_t j{}; j < kernelSize; ++j)
        {
            myKernel[i][j] = randomStartVal();
        }
    }

    factory::Factory factory{};
    myActFunc = factory.actFunc(actFunc);

    // The bias is already initialized in the constructor - no need to do it again.
}

// -----------------------------------------------------------------------------
bool Conv::padInput(const Matrix2d& input) noexcept
{
    initMatrix(myInputPadded);

    for (std::size_t i{}; i < input.size(); ++i)
    {
        if (!isMatrixSquare(input, i)) { return false; }

        for (std::size_t j{}; j < input.size(); ++j)
        {
            const std::size_t offset{myKernel.size() / 2U};
            myInputPadded[i + offset][j + offset] = input[i][j];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Conv::extractUnpaddedGradients() noexcept
{
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            const std::size_t offset{myKernel.size() / 2U};
            myInputGradients[i][j] = myInputGradientsPadded[i + offset][j + offset];
        }
    }
}
} // namespace ml::conv_layer
