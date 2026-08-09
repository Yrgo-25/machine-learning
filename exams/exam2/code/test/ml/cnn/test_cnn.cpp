/**
 * @file Component tests for ml::cnn::Cnn.
 */
#include <cstddef>

#include "ml/cnn/cnn.h"
#include "ml/factory/stub.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

namespace
{
using namespace ml;
using ml::cnn::Cnn;

/**
 * @brief Verify that inputSize() and outputSize() match the given construction parameters,
 *        using an all-stub factory so the test is independent of Conv/MaxPool/Flatten/Dense.
 */
TEST(Cnn, InputAndOutputSizeMatchConstructionParameters)
{
    constexpr std::size_t convInput{4U};
    constexpr std::size_t convKernel{2U};
    constexpr std::size_t poolSize{2U};
    constexpr std::size_t denseOutput{1U};

    factory::Stub factory{};
    Cnn cnn{factory,  convInput,   convKernel,          act_func::Type::Relu,
            poolSize, denseOutput, act_func::Type::Tanh};

    EXPECT_EQ(cnn.inputSize(), convInput);
    EXPECT_EQ(cnn.outputSize(), denseOutput);
}

/**
 * @brief Verify that train() returns false for invalid parameters, true otherwise.
 */
TEST(Cnn, TrainReturnsFalseForInvalidParameters)
{
    constexpr std::size_t convInput{4U};
    constexpr std::size_t convKernel{2U};
    constexpr std::size_t poolSize{2U};
    constexpr std::size_t denseOutput{1U};

    factory::Stub factory{};
    Cnn cnn{factory,  convInput,   convKernel,          act_func::Type::Relu,
            poolSize, denseOutput, act_func::Type::Tanh};

    const Matrix3d trainIn{Matrix2d(convInput, Matrix1d(convInput, 0.0))};
    const Matrix2d trainOut{Matrix1d(denseOutput, 0.0)};

    constexpr std::size_t validEpochCount{1U};
    constexpr std::size_t invalidEpochCount{0U};
    constexpr double validLearningRate{0.1};
    constexpr double invalidLowLearningRate{0.0};
    constexpr double invalidHighLearningRate{1.1};

    EXPECT_FALSE(cnn.train(trainIn, trainOut, invalidEpochCount, validLearningRate));
    EXPECT_FALSE(cnn.train(trainIn, trainOut, validEpochCount, invalidLowLearningRate));
    EXPECT_FALSE(cnn.train(trainIn, trainOut, validEpochCount, invalidHighLearningRate));
    EXPECT_TRUE(cnn.train(trainIn, trainOut, validEpochCount, validLearningRate));
}
} // namespace
