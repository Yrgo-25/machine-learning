/**
 * @file Training and prediction of a CNN (Convolutional Neural Network).
 */
#include <cstddef>
#include <cstdio>

#include "ml/cnn/cnn.h"
#include "ml/factory/factory.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace
{
/**
 * @brief Predict and print with the given CNN (Convolutional Neural Network).
 *
 * @param[in] cnn The CNN with which to predict.
 * @param[in] inputs Input sets to predict with.
 */
void predictAndPrint(ml::cnn::Interface& cnn, const ml::Matrix3d& inputs) noexcept
{
    if (inputs.empty()) { return; }
    const auto last = &inputs[inputs.size() - 1U];

    std::printf(
        "--------------------------------------------------------------------------------\n");

    for (const auto& input : inputs)
    {
        std::printf("Input:\n");
        ml::printMatrix(input);

        std::printf("\nPrediction:\n");
        ml::printMatrix(cnn.predict(input));

        if (&input != last) { std::printf("\n"); }
    }
    std::printf(
        "--------------------------------------------------------------------------------\n\n");
}
} // namespace

/**
 * @brief Create and train a CNN (Convolutional Neural Network). Print predictions on success.
 *
 * @return 0 on success, error code -1 on failure.
 */
int main()
{
    // CNN parameters.
    constexpr std::size_t inputSize{4U};
    constexpr std::size_t kernelSize{2U};
    constexpr ml::act_func::Type convFunc{ml::act_func::Type::Tanh};
    constexpr std::size_t poolSize{2U};
    constexpr std::size_t denseOutput{4U};
    constexpr ml::act_func::Type denseFunc{ml::act_func::Type::Tanh};

    // Training parameters.
    constexpr std::size_t epochCount{100000U};
    constexpr double learningRate{0.01};

    // Input data for training (digits 0 - 3).
    const ml::Matrix3d inputs{
        {{1, 1, 1, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 1, 1}},
        {{0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 0, 0}},
        {{1, 0, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}},
        {{0, 0, 1, 0}, {0, 0, 1, 0}, {1, 1, 1, 1}, {0, 0, 1, 0}},
    };
    // Output data for training (one-hot encoded: the target digit's node is 1.0, the rest -1.0).
    const ml::Matrix2d outputs{
        {1.0, -1.0, -1.0, -1.0},
        {-1.0, 1.0, -1.0, -1.0},
        {-1.0, -1.0, 1.0, -1.0},
        {-1.0, -1.0, -1.0, 1.0},
    };

    auto factory = ml::factory::create();
    ml::cnn::Cnn cnn{*factory, inputSize, kernelSize, convFunc, poolSize, denseOutput, denseFunc};
    const auto success = cnn.train(inputs, outputs, epochCount, learningRate);
    if (success) { predictAndPrint(cnn, inputs); }
    else { std::fprintf(stderr, "Training failed!\n"); }
    return success ? 0 : -1;
}
