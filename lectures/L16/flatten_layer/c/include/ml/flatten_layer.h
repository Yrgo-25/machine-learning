/**
 * @file Simple flatten layer implementation.
 */
#ifndef ML_FLATTEN_LAYER_H_
#define ML_FLATTEN_LAYER_H_

#include <stdbool.h>

#include "ml/matrix.h"

/** Flatten layer structure. */
typedef struct flatten_layer flatten_layer_t;

/**
 * @brief Create and initialize a new flatten layer.
 *
 * @param[in] input_size Input size (assumed square).
 *
 * @return Pointer to the new flatten layer, or nullptr in failure.
 */
flatten_layer_t* flatten_layer_new(size_t input_size);

/**
 * @brief Delete given flatten layer.
 *
 *        Release allocated resources and set the corresponding pointer to null.
 *
 * @param[in] self Double pointer to the flatten layer.
 *
 * @return Pointer to the new flatten layer, or nullptr in failure.
 */
void flatten_layer_del(flatten_layer_t** self);

/**
 * @brief Get flatten layer output.
 *
 * @param[in] self Pointer to the flatten layer.
 *
 * @return Pointer to matrix holding the flatten layer output.
 */
const matrix_t* flatten_layer_output(const flatten_layer_t* self);

/**
 * @brief Get flatten layer input gradients.
 *
 * @param[in] self Pointer to the flatten layer.
 *
 * @return Pointer to matrix holding the flatten layer input gradients.
 */
const matrix_t* flatten_layer_input_gradients(const flatten_layer_t* self);

/**
 * @brief Run feedforward operation.
 *
 * @param[in] self Pointer to the flatten layer.
 * @param[in] input Input data.
 *
 * @return True on success, false on failure.
 */
bool flatten_layer_feedforward(flatten_layer_t* self, const matrix_t* input);

/**
 * @brief Perform backpropagation.
 *
 * @param[in] self Pointer to the flatten layer.
 * @param[in] output_gradients Output gradients from the next layer.
 *
 * @return True on success, false on failure.
 */
bool flatten_layer_backpropagate(flatten_layer_t* self, const matrix_t* output_gradients);

#endif /** ML_FLATTEN_LAYER_H_ */
