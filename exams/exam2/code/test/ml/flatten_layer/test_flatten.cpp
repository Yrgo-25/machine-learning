/**
 * @file Unit tests for ml::flatten_layer::Flatten.
 */
#include <cstddef>

#include "ml/flatten_layer/flatten.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

namespace
{
using ml::Matrix1d;
using ml::Matrix2d;
using ml::flatten_layer::Flatten;

/** Layer dimensions used by the test cases. */
constexpr std::size_t InputSize{4U};

/**
 * @brief Verify that the flatten layer turns a square input into a vector of every element.
 */
TEST(Flatten, ConstructorSetsDimensions)
{
    Flatten layer{InputSize};

    EXPECT_EQ(layer.inputSize(), InputSize);
    EXPECT_EQ(layer.outputSize(), InputSize * InputSize);
    EXPECT_EQ(layer.output().size(), InputSize * InputSize);
}

/**
 * @brief Verify that feedforward rejects input of the wrong size.
 *
 * @note The rejected call prints an error message, which is expected and does not mean that the
 *       test failed.
 */
TEST(Flatten, FeedforwardChecksInputSize)
{
    const Matrix2d wrongSized(InputSize + 1U, Matrix1d(InputSize + 1U, 0.0));
    const Matrix2d validInput(InputSize, Matrix1d(InputSize, 0.0));
    Flatten layer{InputSize};

    EXPECT_FALSE(layer.feedforward(wrongSized));
    EXPECT_TRUE(layer.feedforward(validInput));
}
} // namespace
