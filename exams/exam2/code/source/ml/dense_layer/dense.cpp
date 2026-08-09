/**
 * @file Dense layer implementation details.
 */
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/act_func/type.h"
#include "ml/dense_layer/dense.h"
#include "ml/factory/factory.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Dense::Dense(const std::size_t inputSize, const std::size_t outputSize,
             const act_func::Type actFunc) noexcept
    : myInputGradients{}
    , myBias{}
    , myWeights{}
    , myOutput{}
    , myPreActivationOutput{}
    , myError{}
    , myActFunc{nullptr}
{
    checkParameters(inputSize, outputSize);
    init(inputSize, outputSize, actFunc);
}

// -----------------------------------------------------------------------------
std::size_t Dense::inputSize() const noexcept
{
    return myWeights.empty() ? 0U : myWeights[0U].size();
}

// -----------------------------------------------------------------------------
std::size_t Dense::outputSize() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
const std::vector<double>& Dense::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const std::vector<double>& Dense::inputGradients() const noexcept { return myInputGradients; }

// -----------------------------------------------------------------------------
bool Dense::feedforward(const std::vector<double>& input) noexcept
{
    if (!matchDimensions(inputSize(), input.size())) { return false; }

    for (std::size_t i{}; i < outputSize(); ++i)
    {
        double sum{myBias[i]};

        for (std::size_t j{}; j < inputSize(); ++j)
        {
            sum += myWeights[i][j] * input[j];
        }
        myPreActivationOutput[i] = sum;
        myOutput[i]              = myActFunc->output(sum);
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const std::vector<double>& outputGradients) noexcept
{
    if (!matchDimensions(outputSize(), outputGradients.size())) { return false; }

    for (std::size_t i{}; i < outputSize(); ++i)
    {
        const double error{outputGradients[i] - myOutput[i]};
        myError[i] = error * myActFunc->delta(myPreActivationOutput[i]);
    }

    initMatrix(myInputGradients);

    for (std::size_t i{}; i < inputSize(); ++i)
    {
        for (std::size_t j{}; j < outputSize(); ++j)
        {
            myInputGradients[i] += myError[j] * myWeights[j][i];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::optimize(const std::vector<double>& input, const double learningRate) noexcept
{
    if (!matchDimensions(inputSize(), input.size()) || (!checkLearningRate(learningRate)))
    {
        return false;
    }

    for (std::size_t i{}; i < outputSize(); ++i)
    {
        myBias[i] += myError[i] * learningRate;

        for (std::size_t j{}; j < inputSize(); ++j)
        {
            myWeights[i][j] += myError[i] * learningRate * input[j];
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Dense::checkParameters(const std::size_t inputSize, const std::size_t outputSize) noexcept
{
    if (0U == outputSize)
    {
        std::fprintf(stderr, "Node count cannot be 0!\n");
        std::terminate();
    }
    else if (0U == inputSize)
    {
        std::fprintf(stderr, "Weight count cannot be 0!\n");
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
void Dense::init(const std::size_t inputSize, const std::size_t outputSize,
                 const act_func::Type actFunc) noexcept
{
    initMatrix(myInputGradients, inputSize);
    initMatrix(myBias, outputSize);
    initMatrix(myWeights, outputSize, inputSize);
    initMatrix(myOutput, outputSize);
    initMatrix(myPreActivationOutput, outputSize);
    initMatrix(myError, outputSize);

    for (std::size_t i{}; i < outputSize; ++i)
    {
        myBias[i] = randomStartVal();

        for (std::size_t j{}; j < inputSize; ++j)
        {
            myWeights[i][j] = randomStartVal();
        }
    }
    factory::Factory factory{};
    myActFunc = factory.actFunc(actFunc);
}
} // namespace ml::dense_layer
