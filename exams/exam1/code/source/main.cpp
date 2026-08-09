/**
 * @file Neural network implementation.
 */
#include <cstdint>
#include <cstdio>

#include "ml/neural_network/shallow.h"
#include "ml/types.h"

#ifdef STUB
#include "ml/dense_layer/stub.h"
using DenseLayer = ml::dense_layer::Stub;
#else
#include "ml/dense_layer/dense.h"
using DenseLayer = ml::dense_layer::Dense;
#endif // STUB

namespace
{
// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix1d& matrix, const char* end = "\n") noexcept
{
    const auto size = static_cast<std::uint16_t>(matrix.size());
    std::printf("{");

    if (0U < size)
    {
        const auto last = &(matrix[matrix.size() - 1U]);

        for (auto it = matrix.data(); it < last; ++it)
        {
            std::printf("%.1f, ", *it);
        }
        std::printf("%.1f", *last);
    }
    std::printf("}%s", nullptr != end ? end : "");
}

// -----------------------------------------------------------------------------
void testNetwork(ml::INeuralNetwork& network, const ml::Matrix2d& inputs) noexcept
{
    for (const auto& input : inputs)
    {
        const auto& prediction = network.predict(input);
        std::printf("Input: ");
        printMatrix(input, ", ");
        std::printf("predicted output: ");
        printMatrix(prediction, "\n");
    }
}
} // namespace

/**
 * @brief Application entry point.
 *
 * @return 0 on termination of the program.
 */
int main()
{
    constexpr std::uint16_t inputCount{2U};
    constexpr std::uint16_t hiddenCount{3U};
    constexpr std::uint16_t outputCount{1U};

    constexpr std::uint32_t epochCount{20000U};
    constexpr double learningRate{0.1};

    const ml::Matrix2d trainIn{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const ml::Matrix2d trainOut{{0.0}, {1.0}, {1.0}, {0.0}};

    // Create dense layers (for 2-3-1 network).
    DenseLayer hiddenLayer{hiddenCount, inputCount, ml::ActFunc::Relu};
    DenseLayer outputLayer{outputCount, hiddenCount, ml::ActFunc::Tanh};

    // Create and train shallow neural network.
    ml::neural_network::Shallow network{hiddenLayer, outputLayer};
    network.train(trainIn, trainOut, epochCount, learningRate);
    testNetwork(network, trainIn);
    return 0;
}
