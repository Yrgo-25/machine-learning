/**
 * @file Dense layer demo.
 */
#include <cstddef>
#include <iostream>

#include "ml/dense_layer/stub.h"
#include "ml/types.h"

namespace
{
// -----------------------------------------------------------------------------
void printSeparator() noexcept
{
    constexpr std::size_t separatorLength{80U};

    for (std::size_t i{}; i < separatorLength; ++i)
    {
        std::cout << '-';
    }
    std::cout << '\n';
}

// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix1d& matrix) noexcept
{
    for (std::size_t i{}; i < matrix.size(); ++i)
    {
        if (0U < i) { std::cout << ' '; }
        std::cout << matrix[i];
    }
    std::cout << '\n';
}

// -----------------------------------------------------------------------------
void printLayerInfo(const char* name, const ml::dense_layer::Interface& denseLayer) noexcept
{
    std::cout << name << ": node count: " << denseLayer.nodeCount()
              << ", weight count: " << denseLayer.weightCount() << ", output: ";
    printMatrix(denseLayer.output());
}
} // namespace

/**
 * @brief Application entry point.
 *
 * @return 0 on successful termination.
 */
int main()
{
    constexpr std::size_t inputCount{2U};
    constexpr std::size_t hiddenCount{3U};
    constexpr std::size_t outputCount{1U};
    constexpr double outputLayerValue{0.8};
    constexpr double inputValue{1.0};
    constexpr double referenceValue{1.0};
    constexpr double validLearningRate{0.01};
    constexpr double invalidLearningRate{1.5};
    constexpr double newOutputValue{0.1};

    // Create the hidden layer with the default output value, and the output layer with its own.
    // The output layer holds one weight per hidden node, which is what connects the two.
    ml::dense_layer::Stub hiddenLayer{hiddenCount, inputCount};
    ml::dense_layer::Stub outputLayer{outputCount, hiddenCount, outputLayerValue};

    // Inputs for the hidden layer: one of the right size, and one a single element too long.
    const ml::Matrix1d validInput(inputCount, inputValue);
    const ml::Matrix1d invalidInput(inputCount + 1U, inputValue);
    const ml::Matrix1d reference(outputCount, referenceValue);

    // Print return values as true/false rather than 1/0.
    std::cout << std::boolalpha;

    // Print the dimensions and the output of both layers.
    printSeparator();
    printLayerInfo("Hidden layer", hiddenLayer);
    printLayerInfo("Output layer", outputLayer);
    printSeparator();

    // Feed the hidden layer twice; the second call is rejected but still counted.
    const auto validFeedforward   = hiddenLayer.feedforward(validInput);
    const auto invalidFeedforward = hiddenLayer.feedforward(invalidInput);
    std::cout << "Feedforward with " << validInput.size() << " inputs: " << validFeedforward
              << '\n';
    std::cout << "Feedforward with " << invalidInput.size() << " inputs: " << invalidFeedforward
              << '\n';
    std::cout << "Output after feedforward: ";
    printMatrix(hiddenLayer.output());
    std::cout << "Feedforward count: " << hiddenLayer.feedforwardCount() << '\n';
    printSeparator();

    // Backpropagate both layers, then optimize the hidden layer with a valid and an invalid rate.
    std::cout << "Backpropagate output layer: " << outputLayer.backpropagate(reference) << '\n';
    std::cout << "Backpropagate hidden layer: " << hiddenLayer.backpropagate(outputLayer) << '\n';
    std::cout << "Optimize with learning rate " << validLearningRate << ": "
              << hiddenLayer.optimize(validInput, validLearningRate) << '\n';
    std::cout << "Optimize with learning rate " << invalidLearningRate << ": "
              << hiddenLayer.optimize(validInput, invalidLearningRate) << '\n';
    printSeparator();

    // Drive the hidden layer's output and reset its feedforward count, as the tests in L07 do.
    hiddenLayer.setOutput(newOutputValue);
    hiddenLayer.clearFeedforwardCount();
    std::cout << "Output after setOutput(" << newOutputValue << "): ";
    printMatrix(hiddenLayer.output());
    std::cout << "Feedforward count after clearFeedforwardCount(): "
              << hiddenLayer.feedforwardCount() << '\n';
    printSeparator();
    return 0;
}
