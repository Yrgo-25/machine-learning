/**
 * @file Neural network demo.
 */
#include <cstddef>
#include <cstdio>

#include "ml/dense_layer/stub.h"
#include "ml/neural_network/interface.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"

namespace
{
// -----------------------------------------------------------------------------
void printSeparator() noexcept
{
    constexpr std::size_t separatorLength{80U};

    for (std::size_t i{}; i < separatorLength; ++i)
    {
        std::printf("-");
    }
    std::printf("\n");
}

// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix1d& matrix) noexcept
{
    for (std::size_t i{}; i < matrix.size(); ++i)
    {
        if (0U < i) { std::printf(" "); }
        std::printf("%g", matrix[i]);
    }
}

// -----------------------------------------------------------------------------
void printPredictions(const char* header, ml::neural_network::Interface& network,
                      const ml::Matrix2d& trainInput) noexcept
{
    std::printf("%s:\n", header);

    for (const auto& input : trainInput)
    {
        std::printf("Input: ");
        printMatrix(input);
        std::printf(", predicted output: ");
        printMatrix(network.predict(input));
        std::printf("\n");
    }
}

// -----------------------------------------------------------------------------
bool trainAndTest(ml::neural_network::Shallow& network, const ml::Matrix2d& trainInput) noexcept
{
    constexpr std::size_t epochCount{100U};
    constexpr double learningRate{0.01};

    // Predict before training.
    printSeparator();
    printPredictions("Predictions before training", network, trainInput);
    printSeparator();

    if (!network.train(epochCount, learningRate))
    {
        std::fprintf(stderr, "Training failed!\n");
        return false;
    }
    printPredictions("Predictions after training", network, trainInput);
    printSeparator();
    return true;
}
} // namespace

/**
 * @brief Application entry point.
 *
 * @return 0 on success, -1 if training fails.
 */
int main()
{
    constexpr std::size_t inputCount{2U};
    constexpr std::size_t hiddenCount{3U};
    constexpr std::size_t outputCount{1U};

    // Training data: the 2-bit XOR pattern.
    const ml::Matrix2d trainIn{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const ml::Matrix2d trainOut{{0.0}, {1.0}, {1.0}, {0.0}};

    // Create the two layers, then the network they form. The output layer holds one weight per
    // hidden node, which is what connects the two.
    ml::dense_layer::Stub hiddenLayer{hiddenCount, inputCount};
    ml::dense_layer::Stub outputLayer{outputCount, hiddenCount};
    ml::neural_network::Shallow network{hiddenLayer, outputLayer, trainIn, trainOut};

    // Train and test the model.
    return trainAndTest(network, trainIn) ? 0 : -1;
}
