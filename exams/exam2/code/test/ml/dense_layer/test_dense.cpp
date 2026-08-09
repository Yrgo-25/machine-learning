/**
 * @file Unit tests for ml::dense_layer::Dense.
 */
#include <cstddef>

#include "ml/act_func/interface.h"
#include "ml/dense_layer/dense.h"
#include "yrgo/test/test.h"

namespace
{
using namespace ml::dense_layer;

/**
 * @brief Example test: verify that inputSize() and outputSize() match the values provided
 *        at construction. Add further TEST(...) cases below, e.g. for
 *        ml::conv_layer::Conv, ml::conv_layer::MaxPool and ml::flatten_layer::Flatten.
 */
TEST(Dense, ConstructorSetsDimensions)
{
    constexpr std::size_t inputSize{4U};
    constexpr std::size_t outputSize{2U};

    Dense layer{inputSize, outputSize};
    EXPECT_EQ(layer.inputSize(), inputSize);
    EXPECT_EQ(layer.outputSize(), outputSize);
}
} // namespace
