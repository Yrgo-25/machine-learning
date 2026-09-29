/**
 * @file Unit tests for ml::conv_layer::Conv and ml::conv_layer::MaxPool.
 */
#include <cstddef>

#include "ml/act_func/type.h"
#include "ml/conv_layer/conv.h"
#include "ml/conv_layer/max_pool.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

namespace
{
using ml::Matrix1d;
using ml::Matrix2d;
using ml::act_func::Type;
using ml::conv_layer::Conv;
using ml::conv_layer::MaxPool;

/** Layer dimensions used by the test cases. */
constexpr std::size_t InputSize{4U};
constexpr std::size_t KernelSize{3U};
constexpr std::size_t PoolSize{2U};

/**
 * @brief Verify that the convolutional layer keeps the spatial size of its input, which is what
 *        the zero padding is there for, and that the kernel has the requested size.
 */
TEST(Conv, ConstructorSetsDimensions)
{
    Conv layer{InputSize, KernelSize, Type::Relu};

    EXPECT_EQ(layer.inputSize(), InputSize);
    EXPECT_EQ(layer.outputSize(), InputSize);
    EXPECT_EQ(layer.kernel().size(), KernelSize);
    EXPECT_EQ(layer.kernel()[0U].size(), KernelSize);
}

/**
 * @brief Verify that the pooling layer divides the spatial size by the pool size.
 */
TEST(MaxPool, ConstructorSetsDimensions)
{
    MaxPool layer{InputSize, PoolSize};

    EXPECT_EQ(layer.inputSize(), InputSize);
    EXPECT_EQ(layer.outputSize(), InputSize / PoolSize);
}

/**
 * @brief Verify that feedforward rejects input of the wrong size.
 *
 * @note The rejected call prints an error message, which is expected and does not mean that the
 *       test failed.
 */
TEST(Conv, FeedforwardChecksInputSize)
{
    const Matrix2d wrongSized(InputSize + 1U, Matrix1d(InputSize + 1U, 0.0));
    const Matrix2d validInput(InputSize, Matrix1d(InputSize, 0.0));
    Conv layer{InputSize, KernelSize, Type::Relu};

    EXPECT_FALSE(layer.feedforward(wrongSized));
    EXPECT_TRUE(layer.feedforward(validInput));
}
} // namespace
