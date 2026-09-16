/**
 * @file Unit tests for ml::dense_layer::Stub.
 */
#include <cstddef>
#include <type_traits>

#include "ml/dense_layer/interface.h"
#include "ml/dense_layer/stub.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

using DenseLayer          = ml::dense_layer::Stub;
using DenseLayerInterface = ml::dense_layer::Interface;
using Matrix1d            = ml::Matrix1d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance when using EXPECT_NEAR(). */
    static constexpr double Tolerance{1e-9};

    /** Node count used by most tests. */
    static constexpr std::size_t NodeCount{3U};

    /** Weight count per node used by most tests. */
    static constexpr std::size_t WeightCount{2U};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};
};

/**
 * @brief Check whether a layer can be implicitly constructed from a node count and a weight count,
 *        i.e. via copy-list-initialization such as `const DenseLayer& layer = {3U, 2U};`.
 *
 *        This overload is selected when the constructor is not explicit.
 *
 * @return True.
 */
template<typename T>
constexpr auto isImplicitlyConstructible(int) noexcept
    -> decltype(std::declval<void (*)(const T&)>()({std::size_t{}, std::size_t{}}), bool{})
{
    return true;
}

/**
 * @brief Check whether a layer can be implicitly constructed from a node count and a weight count.
 *
 *        This overload is selected when the constructor is explicit.
 *
 * @return False.
 */
template<typename T>
constexpr bool isImplicitlyConstructible(...) noexcept
{
    return false;
}

/**
 * @brief Verify that the layer reports the node and weight counts it was constructed with.
 */
TEST(DenseLayerStub, ConstructedDimensions)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.nodeCount(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weightCount(), Test::WeightCount);
}

/**
 * @brief Verify that the output holds one element per node, each set to default output value 0.5.
 */
TEST(DenseLayerStub, DefaultOutputValue)
{
    constexpr double defaultOutputValue{0.5};
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.output().size(), Test::NodeCount);

    for (const auto output : denseLayer.output())
    {
        EXPECT_NEAR(output, defaultOutputValue, Test::Tolerance);
    }
}

/**
 * @brief Verify that an explicitly given output value is used instead of the default, which is
 *        what lets two layers in the same network be told apart.
 */
TEST(DenseLayerStub, ExplicitOutputValue)
{
    constexpr double outputValues[]{0.25, -1.5, 0.0};

    for (const auto outputValue : outputValues)
    {
        const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, outputValue};

        for (const auto output : denseLayer.output())
        {
            EXPECT_NEAR(output, outputValue, Test::Tolerance);
        }
    }
}

/**
 * @brief Verify that setOutput() replaces every value in the output, and that a reference taken
 *        before the call sees the change.
 */
TEST(DenseLayerStub, SetOutput)
{
    constexpr double initialValue{0.5};
    constexpr double newValues[]{2.0, -1.0, 0.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, initialValue};
    const Matrix1d& output{denseLayer.output()};

    // Test setting the output, reading it back through a reference taken before the call.
    // Expect every value to change, and the node count to stay put.
    for (const auto newValue : newValues)
    {
        denseLayer.setOutput(newValue);

        EXPECT_EQ(output.size(), Test::NodeCount);

        for (const auto value : output)
        {
            EXPECT_NEAR(value, newValue, Test::Tolerance);
        }
    }
}

/**
 * @brief Verify that the error holds one zero per node.
 */
TEST(DenseLayerStub, ErrorIsZero)
{
    constexpr double zero{0.0};

    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_EQ(denseLayer.error().size(), Test::NodeCount);

    for (const auto error : denseLayer.error())
    {
        EXPECT_NEAR(error, zero, Test::Tolerance);
    }
}

/**
 * @brief Verify that the weights form a matrix of one row per node, each row holding only zeros.
 */
TEST(DenseLayerStub, WeightsAreZero)
{
    constexpr double zero{0.0};
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.weights().size(), Test::NodeCount);

    for (const auto& nodeWeights : denseLayer.weights())
    {
        EXPECT_EQ(nodeWeights.size(), Test::WeightCount);

        for (const auto weight : nodeWeights)
        {
            EXPECT_NEAR(weight, zero, Test::Tolerance);
        }
    }
}

/**
 * @brief Verify that feedforward() accepts an input holding one value per weight, and rejects any
 *        other size.
 */
TEST(DenseLayerStub, FeedforwardChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    // Test feedforward with an input holding one value per weight.
    // Expect it to be accepted.
    const Matrix1d input(Test::WeightCount, 1.0);
    EXPECT_TRUE(denseLayer.feedforward(input));

    // Test feedforward with too few and too many values, including an input sized to the node
    // count, which is the mistake this check exists to catch.
    // Expect each of them to be rejected.
    const Matrix1d tooFew(Test::WeightCount - 1U, 1.0);
    const Matrix1d tooMany(Test::WeightCount + 1U, 1.0);
    const Matrix1d nodeSized(Test::NodeCount, 1.0);
    const Matrix1d empty{};

    EXPECT_FALSE(denseLayer.feedforward(tooFew));
    EXPECT_FALSE(denseLayer.feedforward(tooMany));
    EXPECT_FALSE(denseLayer.feedforward(nodeSized));
    EXPECT_FALSE(denseLayer.feedforward(empty));
}

/**
 * @brief Verify that the output layer's backpropagate() accepts reference values holding one value
 *        per node, and rejects any other size.
 */
TEST(DenseLayerStub, BackpropagateChecksReferenceSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    // Test backpropagation with reference values holding one value per node.
    // Expect them to be accepted.
    const Matrix1d reference(Test::NodeCount, 1.0);
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    // Test backpropagation with too few and too many reference values, including a set sized to
    // the weight count. Expect each of them to be rejected.
    const Matrix1d tooFew(Test::NodeCount - 1U, 1.0);
    const Matrix1d tooMany(Test::NodeCount + 1U, 1.0);
    const Matrix1d weightSized(Test::WeightCount, 1.0);
    const Matrix1d empty{};

    EXPECT_FALSE(denseLayer.backpropagate(tooFew));
    EXPECT_FALSE(denseLayer.backpropagate(tooMany));
    EXPECT_FALSE(denseLayer.backpropagate(weightSized));
    EXPECT_FALSE(denseLayer.backpropagate(empty));
}

/**
 * @brief Verify that the hidden layer's backpropagate() accepts a next layer holding one weight
 *        per node of this layer, and rejects any other shape.
 */
TEST(DenseLayerStub, BackpropagateChecksNextLayerWeightCount)
{
    DenseLayer hiddenLayer{Test::NodeCount, Test::WeightCount};

    // Test backpropagation from a next layer with one weight per node of the hidden layer.
    // Expect it to be accepted.
    const DenseLayer matching{1U, Test::NodeCount};
    EXPECT_TRUE(hiddenLayer.backpropagate(matching));

    // Test backpropagation from next layers with too few and too many weights per node.
    // Expect both to be rejected.
    const DenseLayer tooFew{1U, Test::NodeCount - 1U};
    const DenseLayer tooMany{1U, Test::NodeCount + 1U};

    EXPECT_FALSE(hiddenLayer.backpropagate(tooFew));
    EXPECT_FALSE(hiddenLayer.backpropagate(tooMany));
}

/**
 * @brief Verify that optimize() accepts an input holding one value per weight, and rejects any
 *        other size.
 */
TEST(DenseLayerStub, OptimizeChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    // Test optimization with an input holding one value per weight.
    // Expect it to be accepted.
    const Matrix1d input(Test::WeightCount, 1.0);
    EXPECT_TRUE(denseLayer.optimize(input, Test::LearningRate));

    // Test optimization with too few and too many values.
    // Expect both to be rejected.
    const Matrix1d tooFew(Test::WeightCount - 1U, 1.0);
    const Matrix1d tooMany(Test::WeightCount + 1U, 1.0);

    EXPECT_FALSE(denseLayer.optimize(tooFew, Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(tooMany, Test::LearningRate));
}

/**
 * @brief Verify that optimize() accepts a learning rate inside the open range (0.0, 1.0) and
 *        rejects anything at or outside its bounds, matching the range used in L02 and L04.
 */
TEST(DenseLayerStub, OptimizeChecksLearningRate)
{
    constexpr double validLearningRates[]{0.001, 0.5, 0.999};
    constexpr double invalidLearningRates[]{0.0, -0.1, 1.0, 1.5};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const Matrix1d input(Test::WeightCount, 1.0);

    // Test optimization with learning rates inside the valid range.
    // Expect each of them to be accepted.
    for (const auto learningRate : validLearningRates)
    {
        EXPECT_TRUE(denseLayer.optimize(input, learningRate));
    }

    // Test optimization with learning rates at and outside both bounds.
    // Expect each of them to be rejected.
    for (const auto learningRate : invalidLearningRates)
    {
        EXPECT_FALSE(denseLayer.optimize(input, learningRate));
    }
}

/**
 * @brief Verify that feedforward() counts every call it receives.
 */
TEST(DenseLayerStub, FeedforwardCount)
{
    constexpr std::size_t zero{0U};
    constexpr std::size_t one{1U};
    constexpr std::size_t acceptedCallCount{3U};
    constexpr std::size_t rejectedCallCount{2U};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const Matrix1d input(Test::WeightCount, 1.0);
    const Matrix1d wrongSized(Test::WeightCount + 1U, 1.0);
    const Matrix1d reference(Test::NodeCount, 1.0);

    // Test the count on a freshly constructed layer.
    // Expect it to start at zero.
    EXPECT_EQ(denseLayer.feedforwardCount(), zero);

    // Test the count after several accepted calls.
    // Expect one count per call.
    for (std::size_t i{}; i < acceptedCallCount; ++i)
    {
        EXPECT_TRUE(denseLayer.feedforward(input));
    }
    EXPECT_EQ(denseLayer.feedforwardCount(), acceptedCallCount);

    // Test the count after calls rejected for a wrong input size.
    // Expect those to count too: the tally records how often the layer was asked, not how often
    // it agreed.
    for (std::size_t i{}; i < rejectedCallCount; ++i)
    {
        EXPECT_FALSE(denseLayer.feedforward(wrongSized));
    }
    EXPECT_EQ(denseLayer.feedforwardCount(), acceptedCallCount + rejectedCallCount);

    // Test the count after backpropagation and optimization.
    // Expect it untouched, since only feedforward() is counted.
    EXPECT_TRUE(denseLayer.backpropagate(reference));
    EXPECT_TRUE(denseLayer.optimize(input, Test::LearningRate));
    EXPECT_EQ(denseLayer.feedforwardCount(), acceptedCallCount + rejectedCallCount);

    // Test the count after clearing it.
    // Expect it back at zero, and counting up again from there.
    denseLayer.clearFeedforwardCount();
    EXPECT_EQ(denseLayer.feedforwardCount(), zero);

    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_EQ(denseLayer.feedforwardCount(), one);
}

/**
 * @brief Verify that the stub really does compute nothing.
 */
TEST(DenseLayerStub, ComputationLeavesStateUnchanged)
{
    constexpr double outputValue{0.25};
    constexpr double zero{0.0};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, outputValue};

    const Matrix1d input(Test::WeightCount, 1.0);
    const Matrix1d reference(Test::NodeCount, 1.0);
    const DenseLayer nextLayer{1U, Test::NodeCount};

    // Test a full feedforward, backpropagate, and optimize cycle with valid arguments.
    // Expect every call to be accepted.
    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));
    EXPECT_TRUE(denseLayer.backpropagate(nextLayer));
    EXPECT_TRUE(denseLayer.optimize(input, Test::LearningRate));

    // Test the layer's state after that cycle.
    // Expect it to be untouched, since the stub performs range checks only.
    for (const auto output : denseLayer.output())
    {
        EXPECT_NEAR(output, outputValue, Test::Tolerance);
    }
    for (const auto error : denseLayer.error())
    {
        EXPECT_NEAR(error, zero, Test::Tolerance);
    }
    for (const auto& nodeWeights : denseLayer.weights())
    {
        for (const auto weight : nodeWeights)
        {
            EXPECT_NEAR(weight, zero, Test::Tolerance);
        }
    }
}

/**
 * @brief Verify that the stub is usable through ml::dense_layer::Interface.
 */
TEST(DenseLayerStub, UsableThroughInterface)
{
    constexpr double outputValue{0.75};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, outputValue};
    DenseLayerInterface& interface{denseLayer};

    // Test the getters through the interface reference.
    // Expect the same values the concrete class reports.
    EXPECT_EQ(interface.nodeCount(), denseLayer.nodeCount());
    EXPECT_EQ(interface.weightCount(), denseLayer.weightCount());
    EXPECT_EQ(interface.output().size(), denseLayer.output().size());
    EXPECT_NEAR(interface.output()[0U], outputValue, Test::Tolerance);

    // Test the range checks through the interface reference.
    // Expect them to accept and reject exactly as the concrete class does.
    const Matrix1d input(Test::WeightCount, 1.0);
    const Matrix1d wrongSized(Test::WeightCount + 1U, 1.0);

    EXPECT_TRUE(interface.feedforward(input));
    EXPECT_FALSE(interface.feedforward(wrongSized));
}

/**
 * @brief Verify the class properties required by the specification: public inheritance of the
 *        interface, a final class, a virtual destructor, and no copy or move operations.
 */
TEST(DenseLayerStub, ClassProperties)
{
    // Test the inheritance.
    // Expect the stub to be a final class deriving publicly from the interface.
    EXPECT_TRUE((std::is_base_of_v<DenseLayerInterface, DenseLayer>));
    EXPECT_TRUE((std::is_convertible_v<DenseLayer*, DenseLayerInterface*>));
    EXPECT_TRUE((std::is_final_v<DenseLayer>));

    // Test the destructor of the interface.
    // Expect it to be virtual, so a layer deleted through an interface pointer is destroyed fully.
    EXPECT_TRUE((std::has_virtual_destructor_v<DenseLayerInterface>));

    // Test the special member functions.
    // Expect default construction, copying and moving to be unavailable.
    EXPECT_FALSE((std::is_default_constructible_v<DenseLayer>));
    EXPECT_FALSE((std::is_copy_constructible_v<DenseLayer>));
    EXPECT_FALSE((std::is_move_constructible_v<DenseLayer>));
    EXPECT_FALSE((std::is_copy_assignable_v<DenseLayer>));
    EXPECT_FALSE((std::is_move_assignable_v<DenseLayer>));

    // Test the constructor.
    // Expect it to be explicit (no implicit construction from a brace-enclosed list) and noexcept.
    EXPECT_FALSE(isImplicitlyConstructible<DenseLayer>(0));
    EXPECT_TRUE((std::is_nothrow_constructible_v<DenseLayer, std::size_t, std::size_t>));
    EXPECT_TRUE((std::is_nothrow_constructible_v<DenseLayer, std::size_t, std::size_t, double>));
}
} // namespace
