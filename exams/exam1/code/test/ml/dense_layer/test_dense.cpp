/**
 * @file Unit tests for ml::dense_layer::Dense.
 */
#include <cstdint>

#include "ml/dense_layer/dense.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

namespace
{
using ml::ActFunc;
using ml::Matrix1d;
using ml::dense_layer::Dense;

/** Tolerance for comparisons of computed floating-point values. */
constexpr double Tolerance{1e-9};

/** Layer dimensions used by the test cases. */
constexpr std::uint16_t NodeCount{3U};
constexpr std::uint16_t WeightCount{2U};
constexpr std::uint16_t SingleNodeCount{1U};

/** Index of the first node of a layer. */
constexpr std::uint16_t FirstNode{0U};

/**
 * @brief Verify that nodeCount() and weightCount() return the values provided at construction.
 */
TEST(Dense, ConstructorSetsDimensions)
{
    Dense layer{NodeCount, WeightCount};

    EXPECT_EQ(layer.nodeCount(), NodeCount);
    EXPECT_EQ(layer.weightCount(), WeightCount);
    EXPECT_EQ(layer.output().size(), NodeCount);
    EXPECT_EQ(layer.weights().size(), NodeCount);
    EXPECT_EQ(layer.weights()[FirstNode].size(), WeightCount);
}

/**
 * @brief Verify that feedforward rejects input of the wrong size and accepts input of the
 *        expected size.
 *
 * @note The rejected calls print an error message, which is expected and does not mean that the
 *       test failed.
 */
TEST(Dense, FeedforwardChecksInputSize)
{
    const Matrix1d tooSmallInput(WeightCount - 1U, 1.0);
    const Matrix1d tooLargeInput(WeightCount + 1U, 1.0);
    const Matrix1d validInput(WeightCount, 1.0);

    Dense layer{NodeCount, WeightCount};

    EXPECT_FALSE(layer.feedforward(tooSmallInput));
    EXPECT_FALSE(layer.feedforward(tooLargeInput));
    EXPECT_TRUE(layer.feedforward(validInput));
}

/**
 * @brief Verify that feedforward computes bias + sum(weight * input) for a single node.
 *
 *        The start values are random, so the expected value is computed from the layer itself:
 *            - bias() and weights() provide the terms of the weighted sum.
 *            - ActFunc::None passes that sum straight through, without filtering.
 *
 *        This is the pattern to reuse when testing the remaining computations of the layer.
 */
TEST(Dense, FeedforwardComputesWeightedSum)
{
    const Matrix1d input{1.0, 1.0};

    Dense layer{SingleNodeCount, WeightCount, ActFunc::None};

    // Compute the expected output by hand, from the parameters of the layer itself.
    auto expected = layer.bias()[FirstNode];

    for (std::uint16_t j{}; j < layer.weightCount(); ++j)
    {
        expected += layer.weights()[FirstNode][j] * input[j];
    }

    EXPECT_TRUE(layer.feedforward(input));
    EXPECT_NEAR(layer.output()[FirstNode], expected, Tolerance);
}
} // namespace
