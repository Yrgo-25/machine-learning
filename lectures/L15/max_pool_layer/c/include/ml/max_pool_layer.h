/**
 * @file Simple max pooling layer implementation.
 */
#ifndef ML_MAX_POOL_LAYER_H_
#define ML_MAX_POOL_LAYER_H_

#include <stdbool.h>

#include "ml/matrix.h"

/** Max pooling layer structure. */
typedef struct max_pool_layer max_pool_layer_t;

/**
 * @brief Create and initialize a new max pooling layer.
 *
 * @param[in] input_size Input size (assumed square).
 * @param[in] pool_size Pool size (assumed square). Must be evenly divisible by the pool size.
 *
 * @return Pointer to the new max pooling layer, or nullptr in failure.
 */
max_pool_layer_t* max_pool_layer_new(size_t input_size, size_t pool_size);

/**
 * @brief Delete given max pooling layer.
 *
 *        Release allocated resources and set the corresponding pointer to null.
 *
 * @param[in] self Double pointer to the max pooling layer.
 *
 * @return Pointer to the new max pooling layer, or nullptr in failure.
 */
void max_pool_layer_del(max_pool_layer_t** self);

/**
 * @brief Get max pooling layer output.
 *
 * @param[in] self Pointer to the max pooling layer.
 *
 * @return Pointer to matrix holding the max pooling layer output.
 */
const matrix_t* max_pool_layer_output(const max_pool_layer_t* self);

/**
 * @brief Get max pooling layer input gradients.
 *
 * @param[in] self Pointer to the max pooling layer.
 *
 * @return Pointer to matrix holding the max pooling layer input gradients.
 */
const matrix_t* max_pool_layer_input_gradients(const max_pool_layer_t* self);

/**
 * @brief Run feedforward operation.
 *
 * @param[in] self Pointer to the max pooling layer.
 * @param[in] input Input data.
 *
 * @return True on success, false on failure.
 */
bool max_pool_layer_feedforward(max_pool_layer_t* self, const matrix_t* input);

/**
 * @brief Perform backpropagation.
 *
 * @param[in] self Pointer to the max pooling layer.
 * @param[in] output_gradients Output gradients from the next layer.
 *
 * @return True on success, false on failure.
 */
bool max_pool_layer_backpropagate(max_pool_layer_t* self, const matrix_t* output_gradients);

#endif /** ML_MAX_POOL_LAYER_H_ */
