/**
 * @file Convolutional neural network (CNN) implementation.
 */
#pragma once

#include <cstddef>

#include "ml/act_func/type.h"
#include "ml/cnn/interface.h"
#include "ml/types.h"

namespace ml::cnn
{
/**
 * @brief Convolutional neural network (CNN) implementation.
 */
class Cnn final : public Interface
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] factory Machine learning factory.
     * @param[in] convInput Convolutional layer input size.
     * @param[in] convKernel Convolutional layer kernel size.
     * @param[in] convFunc Convolutional layer activation function.
     * @param[in] poolSize Pooling layer size.
     * @param[in] denseOutput Dense layer output size.
     * @param[in] denseFunc Dense layer activation function.
     */
    explicit Cnn(factory::Interface& factory, std::size_t convInput, std::size_t convKernel,
                 act_func::Type convFunc, std::size_t poolSize, std::size_t denseOutput,
                 act_func::Type denseFunc) noexcept;

    /**
     * @brief Destructor.
     */
    ~Cnn() noexcept override;

    /**
     * @brief Get the input size of the CNN.
     *
     * @return The input size of the CNN.
     */
    std::size_t inputSize() const noexcept override;

    /**
     * @brief Get the output size of the CNN.
     *
     * @return The output size of the CNN.
     */
    std::size_t outputSize() const noexcept override;

    /**
     * @brief Predict based on the given input.
     *
     * @param[in] input Input for which to predict.
     *
     * @return The predicted output.
     */
    const Matrix1d& predict(const Matrix2d& input) noexcept override;

    /**
     * @brief Add dense layer.
     *
     *        The input size is automatically adjusted in accordance with the previous layer.
     *
     * @param[in] outputSize Output size.
     * @param[in] actFunc Activation function to use.
     */
    void addDenseLayer(std::size_t outputSize, act_func::Type actFunc) noexcept;

    /**
     * @brief Train the network.
     *
     * @param[in] trainIn Training input sets.
     * @param[in] trainOut Training output sets.
     * @param[in] epochCount Number of epochs to train the model.
     * @param[in] learningRate Learning rate to use during training.
     *
     * @return True on success, false on failure.
     */
    bool train(const Matrix3d& trainIn, const Matrix2d& trainOut, std::size_t epochCount,
               double learningRate) noexcept;

    Cnn()                      = delete; // No default constructor.
    Cnn(const Cnn&)            = delete; // No copy constructor.
    Cnn(Cnn&&)                 = delete; // No move constructor.
    Cnn& operator=(const Cnn&) = delete; // No copy assignment.
    Cnn& operator=(Cnn&&)      = delete; // No move constructor.

private:
    const Matrix1d& output() const noexcept;
    const Matrix2d& convOutput() const noexcept;
    std::size_t convOutputSize() const noexcept;

    bool feedforward(const Matrix2d& input) noexcept;
    bool backpropagate(const Matrix1d& output) noexcept;
    bool optimize(double learningRate) noexcept;

    /** List of convolutional layers. */
    ConvLayerList myConvLayers;

    /** List of dense layers. */
    DenseLayerList myDenseLayers;

    /** Flatten layer between the convolutional and dense layers. */
    FlattenLayerPtr myFlattenLayer;

    /** Machine learning factory. */
    factory::Interface& myFactory;
};
} // namespace ml::cnn