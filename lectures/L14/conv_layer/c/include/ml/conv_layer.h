/**
 * @file Simple convolutional layer implementation.
 */
#ifndef ML_CONV_LAYER_H_
#define ML_CONV_LAYER_H_

#include <stdbool.h>

#include "ml/matrix.h"

/** Convolutional layer structure. */
typedef struct conv_layer conv_layer_t;

/**
 * @brief Create and initialize a new convolutional layer.
 *
 * @param[in] input_size Input size (assumed square).
 * @param[in] kernel_size Kernel size (assumed square).
 *
 * @return Pointer to the new convolutional layer, or nullptr in failure.
 */
conv_layer_t* conv_layer_new(size_t input_size, size_t kernel_size);

/**
 * @brief Delete given convolutional layer.
 *
 *        Release allocated resources and set the corresponding pointer to null.
 *
 * @param[in] self Double pointer to the convolutional layer.
 *
 * @return Pointer to the new convolutional layer, or nullptr in failure.
 */
void conv_layer_del(conv_layer_t** self);

/**
 * @brief Get convolutional layer output.
 *
 * @param[in] self Pointer to the convolutional layer.
 *
 * @return Pointer to matrix holding the convolutional layer output.
 */
const matrix_t* conv_layer_output(const conv_layer_t* self);

/**
 * @brief Get convolutional layer input gradients.
 *
 * @param[in] self Pointer to the convolutional layer.
 *
 * @return Pointer to matrix holding the convolutional layer input gradients.
 */
const matrix_t* conv_layer_input_gradients(const conv_layer_t* self);

/**
 * @brief Run feedforward operation.
 *
 * @param[in] self Pointer to the convolutional layer.
 * @param[in] input Input data.
 *
 * @return True on success, false on failure.
 */
bool conv_layer_feedforward(conv_layer_t* self, const matrix_t* input);

/**
 * @brief Perform backpropagation.
 *
 * @param[in] self Pointer to the convolutional layer.
 * @param[in] output_gradients Output gradients from the next layer.
 *
 * @return True on success, false on failure.
 */
bool conv_layer_backpropagate(conv_layer_t* self, const matrix_t* output_gradients);

/**
 * @brief Optimize parameters using computed gradients.
 *
 * @param[in] self Pointer to the convolutional layer.
 * @param[in] learning_rate Learning rate for parameter update.
 *
 * @return True on success, false on failure.
 */
bool conv_layer_optimize(conv_layer_t* self, double learning_rate);

#endif /** ML_CONV_LAYER_H_ */
