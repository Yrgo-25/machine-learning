/**
 * @file Component tests for ml::neural_network::Shallow.
 */
#include <cstdint>

#include "ml/dense_layer/stub.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

namespace
{
using ml::Matrix1d;
using ml::Matrix2d;
using ml::dense_layer::Stub;
using ml::neural_network::Shallow;

/** Tolerance for comparisons of computed floating-point values. */
constexpr double Tolerance{1e-9};

/** Dimensions of the network used by the test cases. */
constexpr std::uint16_t InputCount{2U};
constexpr std::uint16_t HiddenCount{3U};
constexpr std::uint16_t OutputCount{1U};

/** Index of the first node of a layer. */
constexpr std::uint16_t FirstNode{0U};

/** Training parameters within their valid ranges. */
constexpr std::uint32_t EpochCount{10U};
constexpr double LearningRate{0.1};

/** Epoch count outside its valid range. */
constexpr std::uint32_t ZeroEpochCount{0U};

/** Output value assigned to the output layer stub. */
constexpr double KnownOutput{0.5};

/**
 * @brief Verify that the network reports the dimensions of the layers it was built from.
 *
 *        The layers are stubs rather than Dense instances: a component test of the network is
 *        supposed to fail only when the network itself is wrong.
 */
TEST(Shallow, ReportsLayerDimensions)
{
    Stub hiddenLayer{HiddenCount, InputCount};
    Stub outputLayer{OutputCount, HiddenCount};
    Shallow network{hiddenLayer, outputLayer};

    EXPECT_EQ(network.inputSize(), InputCount);
    EXPECT_EQ(network.outputSize(), OutputCount);
}

/**
 * @brief Verify that training is rejected when the epoch count is zero.
 *
 * @note The rejected call prints an error message, which is expected and does not mean that the
 *       test failed.
 */
TEST(Shallow, TrainChecksEpochCount)
{
    const Matrix2d trainIn{{0.0, 0.0}, {1.0, 1.0}};
    const Matrix2d trainOut{{0.0}, {1.0}};

    Stub hiddenLayer{HiddenCount, InputCount};
    Stub outputLayer{OutputCount, HiddenCount};
    Shallow network{hiddenLayer, outputLayer};

    EXPECT_FALSE(network.train(trainIn, trainOut, ZeroEpochCount, LearningRate));
    EXPECT_TRUE(network.train(trainIn, trainOut, EpochCount, LearningRate));
}

/**
 * @brief Example of how a stub is given a known output.
 *
 *        Stub::setOutput() makes the layer return exactly the given values, which is what makes
 *        it possible to test the network against known layer outputs.
 */
TEST(Shallow, StubReturnsTheOutputItWasGiven)
{
    Stub outputLayer{OutputCount, HiddenCount};
    outputLayer.setOutput(Matrix1d{KnownOutput});

    EXPECT_EQ(outputLayer.output().size(), OutputCount);
    EXPECT_NEAR(outputLayer.output()[FirstNode], KnownOutput, Tolerance);
}
} // namespace
