/**
 * @file Unit and component tests for ml::dense_layer::Dense.
 */
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "ml/dense_layer/dense.h"
#include "ml/dense_layer/interface.h"
#include "ml/neural_network/shallow.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

using ActFunc             = ml::ActFunc;
using DenseLayer          = ml::dense_layer::Dense;
using DenseLayerInterface = ml::dense_layer::Interface;
using NeuralNetwork       = ml::neural_network::Shallow;
using Matrix1d            = ml::Matrix1d;
using Matrix2d            = ml::Matrix2d;

namespace
{
/**
 * @brief Test parameters.
 */
struct Test
{
    /** Value tolerance for hand-computed results when using EXPECT_NEAR(). */
    static constexpr double ExactTolerance{1e-9};

    /** Node count used by most tests. */
    static constexpr std::size_t NodeCount{3U};

    /** Weight count per node used by most tests. Deliberately unequal to NodeCount. */
    static constexpr std::size_t WeightCount{2U};

    /** A learning rate inside the valid range. */
    static constexpr double LearningRate{0.1};

    /** Hidden nodes used by the convergence test. */
    static constexpr std::size_t HiddenCount{8U};

    /** Epochs used by the convergence test. */
    static constexpr std::size_t EpochCount{10000U};

    /** Learning rate used by the convergence test. */
    static constexpr double TrainLearningRate{0.1};

    /** Mean absolute error the convergence test must reach. */
    static constexpr double ConvergenceThreshold{0.05};
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
 * @brief Compute ReLU's output, as specified in appendix B.
 *
 * @param[in] input Input value.
 *
 * @return The input if it's greater than zero, otherwise zero.
 */
double reluOutput(const double input) noexcept { return 0.0 < input ? input : 0.0; }

/**
 * @brief Compute the derivative of the hyperbolic tangent, as specified in appendix B.
 *
 * @param[in] input Input value.
 *
 * @return 1 - tanh(input)^2.
 */
double tanhDelta(const double input) noexcept
{
    const auto tanhOutput = std::tanh(input);
    return 1.0 - tanhOutput * tanhOutput;
}

/**
 * @brief Recover the bias of a ReLU layer.
 *
 *        A zero input makes each node's weighted sum equal to its bias. The bias starts inside
 *        [0.0, 1.0], and ReLU passes non-negative values through unchanged, so the output is the
 *        bias itself.
 *
 * @param[in] denseLayer ReLU layer to recover the bias from. Its output is overwritten.
 *
 * @return The bias of each node.
 */
Matrix1d recoverReluBias(DenseLayer& denseLayer) noexcept
{
    const Matrix1d zeroInput(denseLayer.weightCount(), 0.0);
    denseLayer.feedforward(zeroInput);
    return denseLayer.output();
}

/**
 * @brief Recover the bias of a layer without an activation function.
 *
 *        A zero input makes each node's weighted sum equal to its bias, which the layer outputs
 *        unchanged.
 *
 * @param[in] denseLayer Layer to recover the bias from. Its output is overwritten.
 *
 * @return The bias of each node.
 */
Matrix1d recoverNoneBias(DenseLayer& denseLayer) noexcept
{
    const Matrix1d zeroInput(denseLayer.weightCount(), 0.0);
    denseLayer.feedforward(zeroInput);
    return denseLayer.output();
}

/**
 * @brief Recover the bias of a Tanh layer.
 *
 *        A zero input makes the output tanh(bias), which std::atanh() inverts.
 *
 * @param[in] denseLayer Tanh layer to recover the bias from. Its output is overwritten.
 *
 * @return The bias of each node.
 */
Matrix1d recoverTanhBias(DenseLayer& denseLayer) noexcept
{
    const Matrix1d zeroInput(denseLayer.weightCount(), 0.0);
    denseLayer.feedforward(zeroInput);
    Matrix1d bias(denseLayer.nodeCount(), 0.0);

    for (std::size_t i{}; i < denseLayer.nodeCount(); ++i)
    {
        bias[i] = std::atanh(denseLayer.output()[i]);
    }
    return bias;
}

/**
 * @brief Compute a node's weighted sum by hand, i.e. its output before the activation function.
 *
 * @param[in] weights Weights of the layer.
 * @param[in] bias Bias of the layer.
 * @param[in] input Input to the layer.
 * @param[in] node Index of the node.
 *
 * @return The bias of the node plus the weighted sum of the input.
 */
double weightedSum(const Matrix2d& weights, const Matrix1d& bias, const Matrix1d& input,
                   const std::size_t node) noexcept
{
    auto sum = bias[node];

    for (std::size_t i{}; i < input.size(); ++i)
    {
        sum += weights[node][i] * input[i];
    }
    return sum;
}

/**
 * @brief Compute a network's mean absolute error over a data set with one output value per row.
 *
 * @param[in] network Network to evaluate.
 * @param[in] input Input data.
 * @param[in] reference Reference values.
 *
 * @return The mean absolute error.
 */
double meanAbsoluteError(NeuralNetwork& network, const Matrix2d& input,
                         const Matrix2d& reference) noexcept
{
    double sum{};

    for (std::size_t i{}; i < input.size(); ++i)
    {
        sum += std::fabs(reference[i][0U] - network.predict(input[i])[0U]);
    }
    return sum / static_cast<double>(input.size());
}

/**
 * @brief Verify that the layer reports the node and weight counts it was constructed with.
 */
TEST(DenseLayerDense, ConstructedDimensions)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.nodeCount(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weightCount(), Test::WeightCount);
}

/**
 * @brief Verify that the output, error, and weight matrices are built to the right shape.
 */
TEST(DenseLayerDense, ConstructedMatrixShapes)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    EXPECT_EQ(denseLayer.output().size(), Test::NodeCount);
    EXPECT_EQ(denseLayer.error().size(), Test::NodeCount);
    EXPECT_EQ(denseLayer.weights().size(), Test::NodeCount);

    for (const auto& nodeWeights : denseLayer.weights())
    {
        EXPECT_EQ(nodeWeights.size(), Test::WeightCount);
    }
}

/**
 * @brief Verify that the output and error start at zero.
 */
TEST(DenseLayerDense, ConstructedOutputAndErrorAreZero)
{
    const DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.output()[i], 0.0, Test::ExactTolerance);
        EXPECT_NEAR(denseLayer.error()[i], 0.0, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that a layer can be built with each activation function, or with the default.
 */
TEST(DenseLayerDense, ConstructedWithEachActivationFunction)
{
    const DenseLayer defaultLayer{Test::NodeCount, Test::WeightCount};
    const DenseLayer reluLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    const DenseLayer tanhLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
    const DenseLayer noneLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};

    // Test the three enumerators of ActFunc.
    // Expect them to be distinct values.
    EXPECT_TRUE(ActFunc::Relu != ActFunc::Tanh);
    EXPECT_TRUE(ActFunc::Relu != ActFunc::None);
    EXPECT_TRUE(ActFunc::Tanh != ActFunc::None);

    for (const DenseLayer* denseLayer : {&defaultLayer, &reluLayer, &tanhLayer, &noneLayer})
    {
        EXPECT_EQ(denseLayer->nodeCount(), Test::NodeCount);
        EXPECT_EQ(denseLayer->weightCount(), Test::WeightCount);
    }
}

/**
 * @brief Verify that the getters work through a ml::dense_layer::Interface reference.
 */
TEST(DenseLayerDense, UsableThroughInterface)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const DenseLayerInterface& layerInterface{denseLayer};

    EXPECT_EQ(layerInterface.nodeCount(), Test::NodeCount);
    EXPECT_EQ(layerInterface.weightCount(), Test::WeightCount);
    EXPECT_EQ(layerInterface.output().size(), Test::NodeCount);
    EXPECT_EQ(layerInterface.error().size(), Test::NodeCount);
    EXPECT_EQ(layerInterface.weights().size(), Test::NodeCount);
    EXPECT_TRUE(&layerInterface.output() == &denseLayer.output());
}

/**
 * @brief Verify that feedforward requires one input value per weight.
 */
TEST(DenseLayerDense, FeedforwardChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.feedforward(Matrix1d(Test::WeightCount, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::WeightCount - 1U, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::WeightCount + 1U, 1.0)));
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d{}));

    // Test an input sized to the node count rather than the weight count.
    // Expect rejection: mixing the two counts up is the easiest mistake to make here.
    EXPECT_FALSE(denseLayer.feedforward(Matrix1d(Test::NodeCount, 1.0)));
}

/**
 * @brief Verify that output-layer backpropagation requires one reference value per node.
 */
TEST(DenseLayerDense, BackpropagateChecksReferenceSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.backpropagate(Matrix1d(Test::NodeCount, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::NodeCount - 1U, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::NodeCount + 1U, 1.0)));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d{}));
    EXPECT_FALSE(denseLayer.backpropagate(Matrix1d(Test::WeightCount, 1.0)));
}

/**
 * @brief Verify that hidden-layer backpropagation requires one weight per node in the next layer.
 */
TEST(DenseLayerDense, BackpropagateChecksNextLayerWeightCount)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    DenseLayer matching{1U, Test::NodeCount};
    DenseLayer mismatched{1U, Test::NodeCount + 1U};

    EXPECT_TRUE(denseLayer.backpropagate(matching));
    EXPECT_FALSE(denseLayer.backpropagate(mismatched));
}

/**
 * @brief Verify that optimization requires one input value per weight.
 */
TEST(DenseLayerDense, OptimizeChecksInputSize)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};

    EXPECT_TRUE(denseLayer.optimize(Matrix1d(Test::WeightCount, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::WeightCount - 1U, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::WeightCount + 1U, 1.0), Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d{}, Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(Matrix1d(Test::NodeCount, 1.0), Test::LearningRate));
}

/**
 * @brief Verify that optimization requires a learning rate inside the range (0.0, 1.0).
 */
TEST(DenseLayerDense, OptimizeChecksLearningRate)
{
    constexpr double validRates[]{1e-9, 0.01, 0.5, 0.999};
    constexpr double invalidRates[]{0.0, -1e-9, -0.5, 1.0, 1.5, 100.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const Matrix1d input(Test::WeightCount, 1.0);

    for (const auto learningRate : validRates)
    {
        EXPECT_TRUE(denseLayer.optimize(input, learningRate));
    }
    for (const auto learningRate : invalidRates)
    {
        EXPECT_FALSE(denseLayer.optimize(input, learningRate));
    }
}

/**
 * @brief Verify the class properties required by the specification: public inheritance of the
 *        interface, a final class, a virtual destructor, and no default, copy or move operations.
 */
TEST(DenseLayerDense, ClassProperties)
{
    // Test the inheritance.
    // Expect the layer to be a final class deriving publicly from the interface.
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
    // Expect it to be explicit (no implicit construction from a brace-enclosed list) and noexcept,
    // with the activation function as an optional third argument.
    EXPECT_FALSE(isImplicitlyConstructible<DenseLayer>(0));
    EXPECT_TRUE((std::is_nothrow_constructible_v<DenseLayer, std::size_t, std::size_t>));
    EXPECT_TRUE((std::is_nothrow_constructible_v<DenseLayer, std::size_t, std::size_t, ActFunc>));
}

/**
 * @brief Verify that bias and weights start randomized inside the range [0.0, 1.0].
 */
TEST(DenseLayerDense, ConstructedParametersAreRandomized)
{
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    const auto bias = recoverReluBias(denseLayer);
    bool biasVaries{false};

    for (const auto value : bias)
    {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE((0.0 <= value) && (1.0 >= value));
        biasVaries = biasVaries || (value != bias[0U]);
    }

    // Test that the bias values aren't all the same number.
    // Expect variation: a bias vector left at its initial zeros passes the range check above while
    // quietly dropping one of the layer's two trainable parameters.
    EXPECT_TRUE(biasVaries);

    bool weightsVary{false};

    for (const auto& nodeWeights : denseLayer.weights())
    {
        for (const auto weight : nodeWeights)
        {
            EXPECT_TRUE(std::isfinite(weight));
            EXPECT_TRUE((0.0 <= weight) && (1.0 >= weight));
            weightsVary = weightsVary || (weight != denseLayer.weights()[0U][0U]);
        }
    }

    // Test that the weights aren't all the same number.
    // Expect variation: nodes that start identical compute identical errors and stay identical.
    EXPECT_TRUE(weightsVary);
}

/**
 * @brief Verify that two layers built with the same arguments get different random parameters.
 */
TEST(DenseLayerDense, LayersAreIndependentlyRandomized)
{
    const DenseLayer first{Test::NodeCount, Test::WeightCount};
    const DenseLayer second{Test::NodeCount, Test::WeightCount};
    bool differs{false};

    for (std::size_t i{}; (i < Test::NodeCount) && !differs; ++i)
    {
        for (std::size_t j{}; (j < Test::WeightCount) && !differs; ++j)
        {
            differs = (first.weights()[i][j] != second.weights()[i][j]);
        }
    }

    // Expect a difference: re-seeding the random generator for every layer would, within the same
    // second, give every layer the same parameters.
    EXPECT_TRUE(differs);
}

/**
 * @brief Verify that feedforward computes the bias plus the weighted sum of the input.
 */
TEST(DenseLayerDense, FeedforwardComputesWeightedSum)
{
    const Matrix1d input{2.0, 3.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    const auto bias = recoverReluBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(input));

    // Test each node's output against a sum computed by hand from weights() and the bias.
    // Expect an exact match: with a positive input every sum is positive, and ReLU passes it
    // through unchanged.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto expected = weightedSum(denseLayer.weights(), bias, input, i);
        EXPECT_TRUE(std::isfinite(denseLayer.output()[i]));
        EXPECT_NEAR(denseLayer.output()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that feedforward applies the layer's activation function to the weighted sum.
 */
TEST(DenseLayerDense, FeedforwardAppliesActivationFunction)
{
    // Large negative inputs against non-negative weights drive every sum below zero, so this
    // exercises ReLU's clamp rather than just its pass-through branch.
    const Matrix1d negativeInput{-50.0, -50.0};
    const Matrix1d positiveInput{2.0, 3.0};

    // Case 1 - ReLU.
    {
        DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
        const auto bias = recoverReluBias(denseLayer);

        for (const auto& input : {negativeInput, positiveInput})
        {
            EXPECT_TRUE(denseLayer.feedforward(input));

            for (std::size_t i{}; i < Test::NodeCount; ++i)
            {
                const auto sum = weightedSum(denseLayer.weights(), bias, input, i);
                EXPECT_NEAR(denseLayer.output()[i], reluOutput(sum), Test::ExactTolerance);
            }
        }
    }

    // Case 2 - Tanh.
    {
        DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
        const auto bias = recoverTanhBias(denseLayer);

        for (const auto& input : {negativeInput, positiveInput})
        {
            EXPECT_TRUE(denseLayer.feedforward(input));

            for (std::size_t i{}; i < Test::NodeCount; ++i)
            {
                const auto sum = weightedSum(denseLayer.weights(), bias, input, i);
                EXPECT_NEAR(denseLayer.output()[i], std::tanh(sum), Test::ExactTolerance);
            }
        }
    }

    // Case 3 - None.
    {
        DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
        const auto bias = recoverNoneBias(denseLayer);

        for (const auto& input : {negativeInput, positiveInput})
        {
            EXPECT_TRUE(denseLayer.feedforward(input));

            // Expect the weighted sum itself, negative sums included.
            for (std::size_t i{}; i < Test::NodeCount; ++i)
            {
                const auto sum = weightedSum(denseLayer.weights(), bias, input, i);
                EXPECT_NEAR(denseLayer.output()[i], sum, Test::ExactTolerance);
            }
        }
    }
}

/**
 * @brief Verify that a layer built without an activation function uses ReLU.
 */
TEST(DenseLayerDense, DefaultActivationFunctionIsRelu)
{
    const Matrix1d negativeInput{-50.0, -50.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    const auto bias = recoverReluBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(negativeInput));

    // Test the output for sums below zero.
    // Expect ReLU's clamp to zero: Tanh would give negative values, and no activation at all would
    // give the sums themselves.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto sum = weightedSum(denseLayer.weights(), bias, negativeInput, i);
        EXPECT_NEAR(denseLayer.output()[i], reluOutput(sum), Test::ExactTolerance);
    }
}

/**
 * @brief Verify that feeding the same input twice produces the same output.
 */
TEST(DenseLayerDense, FeedforwardIsDeterministic)
{
    const Matrix1d input{0.5, -0.25};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};

    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto firstOutput = denseLayer.output();

    EXPECT_TRUE(denseLayer.feedforward(input));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.output()[i], firstOutput[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that feedforward alone doesn't change any trainable parameter.
 */
TEST(DenseLayerDense, FeedforwardDoesNotTrain)
{
    const Matrix1d input{1.0, 2.0};
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};

    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverReluBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto biasAfter = recoverReluBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(biasAfter[i], biasBefore[i], Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that an output layer computes its error as the reference minus its own output.
 */
TEST(DenseLayerDense, BackpropagateOutputLayerComputesError)
{
    const Matrix1d input{1.5, 0.5};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto output = denseLayer.output();

    EXPECT_TRUE(denseLayer.backpropagate(reference));

    // Test each node's error against the raw deviation.
    // Expect an exact match: the positive input keeps every sum above zero, where ReLU's derivative
    // is 1.0, so nothing scales the error.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.error()[i], reference[i] - output[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that the error is scaled by the activation function's derivative taken at the
 *        pre-activation weighted sum, not at the layer's output.
 */
TEST(DenseLayerDense, BackpropagateUsesPreActivationDerivative)
{
    const Matrix1d input{1.5, -0.5};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
    const auto bias = recoverTanhBias(denseLayer);

    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    // Expect the derivative at the weighted sum: Tanh's derivative at its own output differs, which
    // is the mistake the OBS! note in appendix B warns about.
    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto preActivation = weightedSum(denseLayer.weights(), bias, input, i);
        const auto expected = (reference[i] - denseLayer.output()[i]) * tanhDelta(preActivation);

        EXPECT_TRUE(std::isfinite(expected));
        EXPECT_NEAR(denseLayer.error()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that a layer without an activation function passes the raw error through, since
 *        the derivative of the identity is 1.0 for every weighted sum.
 */
TEST(DenseLayerDense, BackpropagateNoneUsesUnitDerivative)
{
    // Large negative inputs against non-negative weights drive the sums below zero, where ReLU's
    // derivative is 0.0, so a None layer that falls back on ReLU's derivative doesn't pass.
    const Matrix1d input{-50.0, -50.0};
    const Matrix1d reference{1.0, 0.0, -1.0};

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::None};
    EXPECT_TRUE(denseLayer.feedforward(input));
    const auto output = denseLayer.output();

    EXPECT_TRUE(denseLayer.backpropagate(reference));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(denseLayer.error()[i], reference[i] - output[i], Test::ExactTolerance);
    }
}

/**
 * @brief Verify that a hidden layer computes its error from the next layer's error and weights.
 */
TEST(DenseLayerDense, BackpropagateHiddenLayerComputesError)
{
    constexpr std::size_t nextNodeCount{2U};
    const Matrix1d input{1.0, -2.0};
    const Matrix1d nextReference{1.0, -1.0};

    // ActFunc::Tanh rather than ReLU, so this also pins down which value the derivative is taken
    // at. See BackpropagateUsesPreActivationDerivative above.
    DenseLayer hiddenLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
    DenseLayer nextLayer{nextNodeCount, Test::NodeCount, ActFunc::Tanh};
    const auto bias = recoverTanhBias(hiddenLayer);

    // Give the next layer a non-zero error to propagate back, otherwise every expected value
    // below would be zero and a broken implementation would pass.
    EXPECT_TRUE(hiddenLayer.feedforward(input));
    EXPECT_TRUE(nextLayer.feedforward(hiddenLayer.output()));
    EXPECT_TRUE(nextLayer.backpropagate(nextReference));
    EXPECT_TRUE(hiddenLayer.backpropagate(nextLayer));

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        double sum{};

        for (std::size_t j{}; j < nextNodeCount; ++j)
        {
            sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        const auto preActivation = weightedSum(hiddenLayer.weights(), bias, input, i);
        const auto expected      = sum * tanhDelta(preActivation);

        EXPECT_TRUE(std::isfinite(hiddenLayer.error()[i]));
        EXPECT_NEAR(hiddenLayer.error()[i], expected, Test::ExactTolerance);
    }
}

/**
 * @brief Verify that optimization adjusts every bias and weight by the error, scaled by the
 *        learning rate and, for weights, by the corresponding input.
 */
TEST(DenseLayerDense, OptimizeUpdatesBiasAndWeights)
{
    const Matrix1d input{2.0, -1.0};
    const Matrix1d reference{1.0, 0.0, -1.0};

    // ActFunc::Tanh rather than ReLU: an update can push a bias below zero, where ReLU would report
    // it as zero, while std::atanh() recovers a bias of either sign.
    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};
    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    const auto error         = denseLayer.error();
    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverTanhBias(denseLayer);

    EXPECT_TRUE(denseLayer.optimize(input, Test::LearningRate));
    const auto biasAfter = recoverTanhBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        const auto changeRate = error[i] * Test::LearningRate;
        EXPECT_NEAR(biasAfter[i], biasBefore[i] + changeRate, Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j] + changeRate * input[j],
                        Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that a rejected optimization leaves every parameter untouched.
 */
TEST(DenseLayerDense, OptimizeRejectedLeavesParametersUnchanged)
{
    const Matrix1d input{2.0, 1.0};
    const Matrix1d reference{1.0, 0.0, -1.0};
    const Matrix1d wrongSized(Test::WeightCount + 1U, 1.0);

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    EXPECT_TRUE(denseLayer.feedforward(input));
    EXPECT_TRUE(denseLayer.backpropagate(reference));

    const auto weightsBefore = denseLayer.weights();
    const auto biasBefore    = recoverReluBias(denseLayer);

    EXPECT_FALSE(denseLayer.optimize(wrongSized, Test::LearningRate));
    EXPECT_FALSE(denseLayer.optimize(input, 0.0));
    EXPECT_FALSE(denseLayer.optimize(input, 1.0));

    const auto biasAfter = recoverReluBias(denseLayer);

    for (std::size_t i{}; i < Test::NodeCount; ++i)
    {
        EXPECT_NEAR(biasAfter[i], biasBefore[i], Test::ExactTolerance);

        for (std::size_t j{}; j < Test::WeightCount; ++j)
        {
            EXPECT_NEAR(denseLayer.weights()[i][j], weightsBefore[i][j], Test::ExactTolerance);
        }
    }
}

/**
 * @brief Verify that the computations work through a ml::dense_layer::Interface reference.
 */
TEST(DenseLayerDense, ComputesThroughInterface)
{
    const Matrix1d input(Test::WeightCount, 1.0);
    const Matrix1d reference(Test::NodeCount, 1.0);

    DenseLayer denseLayer{Test::NodeCount, Test::WeightCount};
    DenseLayerInterface& layerInterface{denseLayer};

    EXPECT_TRUE(layerInterface.feedforward(input));
    EXPECT_TRUE(layerInterface.backpropagate(reference));
    EXPECT_TRUE(layerInterface.optimize(input, Test::LearningRate));
    EXPECT_FALSE(layerInterface.feedforward(Matrix1d{}));
}

/**
 * @brief Verify that a network of two dense layers actually learns, by training it on 2-bit XOR
 *        and comparing its error against its own error before training.
 */
TEST(DenseLayerDense, NetworkLearnsXorPattern)
{
    const Matrix2d trainInput{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const Matrix2d trainOutput{{0.0}, {1.0}, {1.0}, {0.0}};

    DenseLayer hiddenLayer{Test::HiddenCount, trainInput[0U].size(), ActFunc::Tanh};
    DenseLayer outputLayer{1U, Test::HiddenCount, ActFunc::Tanh};
    NeuralNetwork network{hiddenLayer, outputLayer, trainInput, trainOutput};

    const auto errorBeforeTraining = meanAbsoluteError(network, trainInput, trainOutput);
    EXPECT_TRUE(std::isfinite(errorBeforeTraining));

    EXPECT_TRUE(network.train(Test::EpochCount, Test::TrainLearningRate));
    const auto errorAfterTraining = meanAbsoluteError(network, trainInput, trainOutput);
    EXPECT_TRUE(std::isfinite(errorAfterTraining));

    // Test the error against the threshold and against the untrained baseline.
    // Expect both: the threshold alone would pass on a network that started out lucky.
    EXPECT_TRUE(errorAfterTraining < Test::ConvergenceThreshold);
    EXPECT_TRUE(errorAfterTraining < errorBeforeTraining);

    for (std::size_t i{}; i < trainInput.size(); ++i)
    {
        const auto prediction = network.predict(trainInput[i])[0U];
        EXPECT_TRUE(std::isfinite(prediction));
        EXPECT_NEAR(prediction, trainOutput[i][0U], Test::ConvergenceThreshold);
    }
}
} // namespace
