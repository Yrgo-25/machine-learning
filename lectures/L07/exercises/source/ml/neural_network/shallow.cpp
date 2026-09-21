/**
 * @file Shallow neural network implementation details.
 */
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>

#include "ml/dense_layer/interface.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"

namespace ml::neural_network
{
// -----------------------------------------------------------------------------
Shallow::Shallow(dense_layer::Interface& hiddenLayer, dense_layer::Interface& outputLayer,
                 const Matrix2d& trainInput, const Matrix2d& trainOutput) noexcept
    : myHiddenLayer{hiddenLayer}
    , myOutputLayer{outputLayer}
    , myTrainInput{trainInput}
    , myTrainOutput{trainOutput}
    , myTrainSetCount{std::min(trainInput.size(), trainOutput.size())}
{
    if (0U == myTrainSetCount)
    {
        std::fprintf(stderr, "Cannot train network without training sets!\n");
        std::terminate();
    }

    if (myHiddenLayer.nodeCount() != myOutputLayer.weightCount())
    {
        std::fprintf(stderr, "Dimension mismatch between the dense layers!\n");
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
const Matrix1d& Shallow::predict(const Matrix1d& input) noexcept
{
    // Run feedforward through each layer, return the neural network output.
    myHiddenLayer.feedforward(input);
    myOutputLayer.feedforward(myHiddenLayer.output());
    return myOutputLayer.output();
}

// -----------------------------------------------------------------------------
bool Shallow::train(const std::size_t epochCount, const double learningRate) noexcept
{
    // Check input arguments, return false if invalid.
    if (0U == epochCount) { return false; }
    const auto lrValid = ((0.0 < learningRate) && (1.0 > learningRate));
    if (!lrValid) { return false; }

    // Run 'epochCount' epochs via a loop.
    for (std::size_t epoch{}; epoch < epochCount; ++epoch)
    {
        // Iterate through all training sets one by one.
        for (std::size_t i{}; i < myTrainSetCount; ++i)
        {
            const auto& input  = myTrainInput[i];
            const auto& output = myTrainOutput[i];

            if (!feedforward(input)) { return false; }
            if (!backpropagate(output)) { return false; }
            if (!optimize(input, learningRate)) { return false; }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool Shallow::feedforward(const Matrix1d& input) noexcept
{
    if (!myHiddenLayer.feedforward(input)) { return false; }

    // The hidden layer's output is the output layer's input.
    return myOutputLayer.feedforward(myHiddenLayer.output());
}

// -----------------------------------------------------------------------------
bool Shallow::backpropagate(const Matrix1d& output) noexcept
{
    if (!myOutputLayer.backpropagate(output)) { return false; }

    // Pass the output layer to access its errors and weights during backpropagation.
    return myHiddenLayer.backpropagate(myOutputLayer);
}

// -----------------------------------------------------------------------------
bool Shallow::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    if (!myHiddenLayer.optimize(input, learningRate)) { return false; }

    // The hidden layer's output is the output layer's input, as in feedforward.
    return myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
}
} // namespace ml::neural_network
