/**
 * @file Component tests for ml::neural_network::Shallow.
 */
#include <cstddef>
#include <type_traits>
#include <utility>

#include "ml/dense_layer/interface.h"
#include "ml/dense_layer/stub.h"
#include "ml/neural_network/interface.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

using DenseLayer             = ml::dense_layer::Stub;
using DenseLayerInterface    = ml::dense_layer::Interface;
using NeuralNetwork          = ml::neural_network::Shallow;
using NeuralNetworkInterface = ml::neural_network::Interface;
using Matrix1d               = ml::Matrix1d;
using Matrix2d               = ml::Matrix2d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance when using EXPECT_NEAR(). */
    static constexpr double Tolerance{1e-9};

    /** Number of network inputs, i.e. weights per hidden node. */
    static constexpr std::size_t InputCount{2U};

    /** Number of hidden nodes. */
    static constexpr std::size_t HiddenCount{3U};

    /** Number of output nodes. */
    static constexpr std::size_t OutputCount{1U};

    /** Number of training sets in the XOR pattern below. */
    static constexpr std::size_t SetCount{4U};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.01};

    /** Output value reported by the hidden layer, distinct from the output layer's. */
    static constexpr double HiddenOutputValue{0.25};

    /** Output value reported by the output layer, distinct from the hidden layer's. */
    static constexpr double OutputOutputValue{0.75};
};

/** Training data inputs: the 2-bit XOR pattern. */
const Matrix2d TrainInput{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

/** Training data outputs: the 2-bit XOR pattern. */
const Matrix2d TrainOutput{{0.0}, {1.0}, {1.0}, {0.0}};

/**
 * @brief Check whether a network can be implicitly constructed from its four constructor arguments,
 *        i.e. via copy-list-initialization such as `const NeuralNetwork& network = {...};`.
 *
 *        This overload is selected when the constructor is not explicit.
 *
 * @return True.
 */
template<typename T>
constexpr auto isImplicitlyConstructible(int) noexcept
    -> decltype(std::declval<void (*)(const T&)>()({std::declval<DenseLayerInterface&>(),
                                                    std::declval<DenseLayerInterface&>(),
                                                    std::declval<const Matrix2d&>(),
                                                    std::declval<const Matrix2d&>()}),
                bool{})
{
    return true;
}

/**
 * @brief Check whether a network can be implicitly constructed from its four constructor arguments.
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
 * @brief Verify that the network hands back the output layer's output, not the hidden layer's.
 */
TEST(NeuralNetworkShallow, PredictReturnsOutputLayerOutput)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    const auto& prediction = network.predict(TrainInput[0U]);

    // Test the shape and the value of the prediction.
    // Expect one value per output node, each reporting the output layer's value.
    EXPECT_EQ(prediction.size(), Test::OutputCount);

    for (const auto value : prediction)
    {
        EXPECT_NEAR(value, Test::OutputOutputValue, Test::Tolerance);
    }
}

/**
 * @brief Verify that the prediction is read from the output layer on every call rather than
 *        copied once, by changing the layer's output behind the network's back.
 */
TEST(NeuralNetworkShallow, PredictReadsOutputLayerLive)
{
    constexpr double newOutputValue{-1.5};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], Test::OutputOutputValue, Test::Tolerance);

    // Test a prediction after the output layer was changed through the reference the network
    // holds. Expect the new value: a network that cached the output would still report the old.
    outputLayer.setOutput(newOutputValue);
    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], newOutputValue, Test::Tolerance);
}

/**
 * @brief Verify that the prediction refers to the output layer's own vector rather than a copy.
 */
TEST(NeuralNetworkShallow, PredictReturnsReferenceToOutputLayer)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(&network.predict(TrainInput[0U]) == &outputLayer.output());
}

/**
 * @brief Verify that a prediction feeds both layers exactly once, in one pass through the network.
 */
TEST(NeuralNetworkShallow, PredictFeedsBothLayersOnce)
{
    constexpr std::size_t one{1U};
    constexpr std::size_t predictionCount{3U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    // Test the counts after construction.
    // Expect both at zero: building a network mustn't feed anything forward.
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{});

    // Test the counts after a single prediction.
    // Expect one pass through each layer, and one output value per output node.
    EXPECT_EQ(network.predict(TrainInput[0U]).size(), Test::OutputCount);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), one);
    EXPECT_EQ(outputLayer.feedforwardCount(), one);

    // Test the counts after several further predictions.
    // Expect one pass through each layer per call, and never more than one.
    for (std::size_t i{}; i < predictionCount; ++i)
    {
        EXPECT_EQ(network.predict(TrainInput[i]).size(), Test::OutputCount);
    }
    EXPECT_EQ(hiddenLayer.feedforwardCount(), one + predictionCount);
    EXPECT_EQ(outputLayer.feedforwardCount(), one + predictionCount);
}

/**
 * @brief Verify that predictions work on an untrained network.
 */
TEST(NeuralNetworkShallow, PredictBeforeTraining)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    for (const auto& input : TrainInput)
    {
        EXPECT_NEAR(network.predict(input)[0U], Test::OutputOutputValue, Test::Tolerance);
    }
}

/**
 * @brief Verify that training performs one feedforward per training set per epoch.
 */
TEST(NeuralNetworkShallow, TrainFeedsEveryTrainingSetEveryEpoch)
{
    // 100 epochs gives the 400 feedforwards stated in appendix B.
    constexpr std::size_t epochCounts[]{1U, 2U, 10U, 100U};

    for (const auto epochCount : epochCounts)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
        EXPECT_EQ(outputLayer.feedforwardCount(), epochCount * Test::SetCount);
    }
}

/**
 * @brief Verify that training with an epoch count of zero fails without touching the layers.
 */
TEST(NeuralNetworkShallow, TrainRejectsZeroEpochCount)
{
    constexpr std::size_t zeroEpochs{0U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_FALSE(network.train(zeroEpochs, Test::LearningRate));

    // Test the counts after the rejected call.
    // Expect them untouched: a rejected training run mustn't feed anything forward.
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{});
}

/**
 * @brief Verify that training fails for learning rates outside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainRejectsInvalidLearningRate)
{
    constexpr std::size_t epochCount{5U};
    constexpr double invalidLearningRates[]{0.0, -0.1, -1.0, 1.0, 1.5, 100.0};

    for (const auto learningRate : invalidLearningRates)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        // Test the tally after the rejected call.
        // Expect zero: the learning rate must be checked before training starts, not left to the
        // layers' own range checks halfway through the first training set.
        EXPECT_FALSE(network.train(epochCount, learningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{});
    }
}

/**
 * @brief Verify that training succeeds for learning rates inside the range (0.0, 1.0).
 */
TEST(NeuralNetworkShallow, TrainAcceptsValidLearningRate)
{
    constexpr std::size_t epochCount{2U};
    constexpr double validLearningRates[]{0.001, 0.01, 0.5, 0.999999};

    for (const auto learningRate : validLearningRates)
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, learningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
    }
}

/**
 * @brief Verify that train() can be called with the epoch count alone.
 *
 *        The stubs accept any learning rate inside the valid range, so the default value itself
 *        (0.01) can't be observed here, only that the default lies inside the range.
 */
TEST(NeuralNetworkShallow, TrainDefaultLearningRate)
{
    constexpr std::size_t epochCount{3U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(network.train(epochCount));
    EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * Test::SetCount);
}

/*
 * A network with no complete training set, and one whose layers don't match, are both rejected by
 * the constructor with std::terminate() (see section 4 of appendix B). Neither can be tested here:
 * std::terminate() takes the whole test program down with it.
 */

/**
 * @brief Verify that training only uses complete training sets, i.e. the smaller of the input and
 *        output row counts.
 */
TEST(NeuralNetworkShallow, TrainUsesSmallestTrainingSetCount)
{
    constexpr std::size_t epochCount{10U};
    constexpr std::size_t shortSetCount{2U};
    const Matrix2d shortInput{TrainInput[0U], TrainInput[1U]};
    const Matrix2d shortOutput{TrainOutput[0U], TrainOutput[1U]};

    // Case 1 - Fewer outputs than inputs.
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, shortOutput};

        EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * shortSetCount);
    }

    // Case 2 - Fewer inputs than outputs.
    {
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, shortInput, TrainOutput};

        EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), epochCount * shortSetCount);
    }
}

/**
 * @brief Verify that training stops at the first failed layer call, rather than carrying on with a
 *        miswired network.
 *
 *        Each case fails during the first training set, so a network that stops there has fed the
 *        hidden layer exactly once, while one that carries on feeds it once per set and epoch.
 */
TEST(NeuralNetworkShallow, TrainStopsAtFirstFailedLayerCall)
{
    constexpr std::size_t epochCount{10U};
    constexpr std::size_t one{1U};

    // Case 1 - The training inputs are wider than the hidden layer's weight count.
    {
        const Matrix2d wideInput{
            {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, wideInput, TrainOutput};

        // Expect the hidden layer's optimization to fail.
        EXPECT_FALSE(network.train(epochCount, Test::LearningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), one);
    }

    // Case 2 - The reference values are wider than the output layer's node count.
    {
        const Matrix2d wideOutput{{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
        DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount};
        DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount};
        NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, wideOutput};

        // Expect the output layer's backpropagation to fail.
        EXPECT_FALSE(network.train(epochCount, Test::LearningRate));
        EXPECT_EQ(hiddenLayer.feedforwardCount(), one);
    }
}

/**
 * @brief Verify that predictions still work once training has run.
 */
TEST(NeuralNetworkShallow, PredictAfterTraining)
{
    constexpr std::size_t epochCount{4U};

    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};

    EXPECT_TRUE(network.train(epochCount, Test::LearningRate));
    hiddenLayer.clearFeedforwardCount();
    outputLayer.clearFeedforwardCount();

    // Test a prediction made after training.
    // Expect it to behave exactly as before training, feeding each layer once.
    EXPECT_NEAR(network.predict(TrainInput[0U])[0U], Test::OutputOutputValue, Test::Tolerance);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{1U});
    EXPECT_EQ(outputLayer.feedforwardCount(), std::size_t{1U});
}

/**
 * @brief Verify that the network is usable through ml::neural_network::Interface.
 */
TEST(NeuralNetworkShallow, UsableThroughInterface)
{
    DenseLayer hiddenLayer{Test::HiddenCount, Test::InputCount, Test::HiddenOutputValue};
    DenseLayer outputLayer{Test::OutputCount, Test::HiddenCount, Test::OutputOutputValue};
    NeuralNetwork network{hiddenLayer, outputLayer, TrainInput, TrainOutput};
    NeuralNetworkInterface& networkInterface{network};

    const auto& prediction = networkInterface.predict(TrainInput[0U]);
    EXPECT_EQ(prediction.size(), Test::OutputCount);
    EXPECT_NEAR(prediction[0U], Test::OutputOutputValue, Test::Tolerance);
    EXPECT_EQ(hiddenLayer.feedforwardCount(), std::size_t{1U});
}

/**
 * @brief Verify the class properties required by the specification: public inheritance of the
 *        interface, a final class, a virtual destructor, no default, copy or move operations, and
 *        noexcept methods.
 */
TEST(NeuralNetworkShallow, ClassProperties)
{
    // Test the inheritance.
    // Expect the network to be a final class deriving publicly from the interface.
    EXPECT_TRUE((std::is_base_of_v<NeuralNetworkInterface, NeuralNetwork>));
    EXPECT_TRUE((std::is_convertible_v<NeuralNetwork*, NeuralNetworkInterface*>));
    EXPECT_TRUE((std::is_final_v<NeuralNetwork>));

    // Test the destructor of the interface.
    // Expect it to be virtual, so a network deleted through an interface pointer is destroyed
    // fully.
    EXPECT_TRUE((std::has_virtual_destructor_v<NeuralNetworkInterface>));

    // Test the special member functions.
    // Expect default construction, copying and moving to be unavailable.
    EXPECT_FALSE((std::is_default_constructible_v<NeuralNetwork>));
    EXPECT_FALSE((std::is_copy_constructible_v<NeuralNetwork>));
    EXPECT_FALSE((std::is_move_constructible_v<NeuralNetwork>));
    EXPECT_FALSE((std::is_copy_assignable_v<NeuralNetwork>));
    EXPECT_FALSE((std::is_move_assignable_v<NeuralNetwork>));

    // Test the constructor.
    // Expect it to be explicit (no implicit construction from a brace-enclosed list) and noexcept.
    EXPECT_FALSE(isImplicitlyConstructible<NeuralNetwork>(0));
    EXPECT_TRUE(
        (std::is_nothrow_constructible_v<NeuralNetwork, DenseLayerInterface&, DenseLayerInterface&,
                                         const Matrix2d&, const Matrix2d&>));

    // Test the remaining methods.
    // Expect predict() and train() to be noexcept.
    EXPECT_TRUE(noexcept(std::declval<NeuralNetwork&>().predict(std::declval<const Matrix1d&>())));
    EXPECT_TRUE(noexcept(std::declval<NeuralNetwork&>().train(std::size_t{}, double{})));
}
} // namespace
