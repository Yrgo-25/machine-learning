/**
 * @file Shallow neural network implementation details.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "ml/dense_layer/interface.h"
#include "ml/helpers.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"

namespace ml::neural_network
{
namespace
{
/**
 * @brief Train order structure.
 */
struct TrainOrder
{
    explicit TrainOrder(std::uint16_t size) noexcept
        : myOrder{}
    {
        myOrder.resize(size);

        for (std::uint16_t i{}; i < size; ++i)
        {
            myOrder[i] = i;
        }
        initRandom();
    }

    const std::uint16_t* data() const noexcept { return myOrder.data(); }
    std::uint16_t size() const noexcept { return static_cast<std::uint16_t>(myOrder.size()); }
    const std::uint16_t* begin() const noexcept { return myOrder.data(); }
    const std::uint16_t* end() const noexcept { return myOrder.data() + myOrder.size(); }

    void randomize() noexcept
    {
        for (std::uint16_t i{}; i < size(); ++i)
        {
            const auto r    = std::rand() % size();
            const auto temp = myOrder[i];
            myOrder[i]      = myOrder[r];
            myOrder[r]      = temp;
        }
    }

private:
    /** Train order. */
    std::vector<std::uint16_t> myOrder;
};

// -----------------------------------------------------------------------------
bool inputValid(const std::uint32_t epochCount, const double learningRate,
                const std::uint16_t setCount) noexcept
{
    if (0U == epochCount)
    {
        std::fprintf(stderr, "Failed to train network: epoch count cannot be 0!\n");
        return false;
    }
    if (0.0 >= learningRate)
    {
        std::fprintf(stderr, "Failed to train network: invalid learning rate %g!\n", learningRate);
        return false;
    }
    if (0U == setCount)
    {
        std::fprintf(stderr, "Failed to train network: missing training sets!\n");
        return false;
    }
    return true;
}
} // namespace

// -----------------------------------------------------------------------------
Shallow::Shallow(IDenseLayer& hiddenLayer, IDenseLayer& outputLayer) noexcept
    : myHiddenLayer{hiddenLayer}
    , myOutputLayer{outputLayer}
{
    const auto weightCount = myHiddenLayer.nodeCount();
    const auto nodeCount   = myOutputLayer.weightCount();

    if (weightCount != nodeCount)
    {
        std::fprintf(stderr,
                     "Dimension mismatch: hidden layer node count (%u) must match "
                     "output layer weight count (%u)!\n",
                     weightCount, nodeCount);
        std::terminate();
    }
}

// -----------------------------------------------------------------------------
std::uint16_t Shallow::inputSize() const noexcept { return myHiddenLayer.weightCount(); }

// -----------------------------------------------------------------------------
std::uint16_t Shallow::outputSize() const noexcept { return myOutputLayer.nodeCount(); }

// -----------------------------------------------------------------------------
const Matrix1d& Shallow::predict(const Matrix1d& input) noexcept
{
    feedforward(input);
    return myOutputLayer.output();
}

// -----------------------------------------------------------------------------
bool Shallow::train(const Matrix2d& trainIn, const Matrix2d& trainOut,
                    const std::uint32_t epochCount, const double learningRate) noexcept
{
    const auto setCount = static_cast<std::uint16_t>(std::min(trainIn.size(), trainOut.size()));
    if (!inputValid(epochCount, learningRate, setCount)) { return false; }

    TrainOrder trainOrder{setCount};

    for (std::uint32_t epoch{}; epoch < epochCount; ++epoch)
    {
        trainOrder.randomize();

        for (const auto i : trainOrder)
        {
            const auto& input  = trainIn[i];
            const auto& output = trainOut[i];

            feedforward(input);
            backpropagate(output);
            optimize(input, learningRate);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
void Shallow::feedforward(const Matrix1d& input) noexcept
{
    myHiddenLayer.feedforward(input);
    myOutputLayer.feedforward(myHiddenLayer.output());
}

// -----------------------------------------------------------------------------
void Shallow::backpropagate(const Matrix1d& output) noexcept
{
    myOutputLayer.backpropagate(output);
    myHiddenLayer.backpropagate(myOutputLayer);
}

// -----------------------------------------------------------------------------
void Shallow::optimize(const Matrix1d& input, const double learningRate) noexcept
{
    myHiddenLayer.optimize(input, learningRate);
    myOutputLayer.optimize(myHiddenLayer.output(), learningRate);
}
} // namespace ml::neural_network
