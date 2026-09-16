/**
 * @file Unit tests for ml::dense_layer::Dense, for the parts implemented in L08.
 */
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "ml/dense_layer/dense.h"
#include "ml/dense_layer/interface.h"
#include "ml/types.h"
#include "yrgo/test/test.h"

using ActFunc             = ml::ActFunc;
using DenseLayer          = ml::dense_layer::Dense;
using DenseLayerInterface = ml::dense_layer::Interface;
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
 * @brief Verify that a layer can be built with either activation function, or with the default.
 */
TEST(DenseLayerDense, ConstructedWithEachActivationFunction)
{
    const DenseLayer defaultLayer{Test::NodeCount, Test::WeightCount};
    const DenseLayer reluLayer{Test::NodeCount, Test::WeightCount, ActFunc::Relu};
    const DenseLayer tanhLayer{Test::NodeCount, Test::WeightCount, ActFunc::Tanh};

    // Test the two enumerators of ActFunc.
    // Expect them to be distinct values.
    EXPECT_TRUE(ActFunc::Relu != ActFunc::Tanh);

    for (const DenseLayer* denseLayer : {&defaultLayer, &reluLayer, &tanhLayer})
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
} // namespace
