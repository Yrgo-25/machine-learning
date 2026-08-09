/**
 * @brief Flatten layer implementation details.
 */
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "ml/flatten_layer.h"
#include "ml/matrix.h"

/**
 * @brief Flatten layer structure.
 */
typedef struct flatten_layer
{
    /** Input gradients. */
    matrix_t* input_gradients;

    /** Output data. */
    matrix_t* output;
} flatten_layer_t;

// -----------------------------------------------------------------------------
static size_t flatten_layer_input_size(const flatten_layer_t* self)
{
    return sqrt(matrix_size(self->input_gradients));
}

// -----------------------------------------------------------------------------
flatten_layer_t* flatten_layer_new(const size_t input_size)
{
    // Check the input size, return null if invalid.
    if (0U == input_size) { return NULL; }

    // Create new flatten layer, return null on failure.
    flatten_layer_t* self = (flatten_layer_t*)(malloc(sizeof(flatten_layer_t)));
    if (NULL == self) { return NULL; }

    // Initialize the member variables.
    self->input_gradients = matrix_new(input_size * input_size);
    self->output          = matrix_new(input_size * input_size);

    // Check whether the member variables were initialize correctly, return null on failure.
    if ((NULL == self->input_gradients) || (NULL == self->output))
    {
        flatten_layer_del(&self);
        return NULL;
    }
    // Return the new flatten layer.
    return self;
}

// -----------------------------------------------------------------------------
void flatten_layer_del(flatten_layer_t** self)
{
    // Check if the layer is valid, terminate the function if not.
    if ((NULL == self) || (NULL == *self)) { return; }

    // Free allocated resources and set the associated pointer to null.
    flatten_layer_t* impl = *self;
    matrix_del(&(impl->input_gradients));
    matrix_del(&(impl->output));
    free(impl);
    *self = NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* flatten_layer_output(const flatten_layer_t* self)
{
    return NULL != self ? self->output : NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* flatten_layer_input_gradients(const flatten_layer_t* self)
{
    return NULL != self ? self->input_gradients : NULL;
}

// -----------------------------------------------------------------------------
bool flatten_layer_feedforward(flatten_layer_t* self, const matrix_t* input)
{
    // Check the input parameters, return false on failure.
    if ((NULL == self) || (NULL == input)) { return false; }
    if (matrix_size(input) != matrix_size(self->input_gradients)) { return false; }

    // Note: Flattening is merely simulated here, since the matrix structure is one-dimensional.
    // We pretend that the input is two-dimensional anyway.
    const size_t input_size = flatten_layer_input_size(self);
    const double* indata    = matrix_data_const(input);
    double* output          = matrix_data(self->output);

    // Perform feedforward - flatten the input.
    for (size_t i = 0U; i < input_size; ++i)
    {
        for (size_t j = 0U; j < input_size; ++j)
        {
            output[i * input_size + j] = indata[i * input_size + j];
        }
    }
    // Return true to indicate success.
    return true;
}

// -----------------------------------------------------------------------------
bool flatten_layer_backpropagate(flatten_layer_t* self, const matrix_t* output_gradients)
{
    // Check the input parameters, return false on failure.
    if ((NULL == self) || (NULL == output_gradients)) { return false; }
    if (matrix_size(output_gradients) != matrix_size(self->output)) { return false; }

    // Note: Unflattening is merely simulated here, since the matrix structure is one-dimensional.
    const size_t input_size = flatten_layer_input_size(self);
    const double* gradients = matrix_data_const(output_gradients);
    double* input_gradients = matrix_data(self->input_gradients);

    // Perform backpropagation - feed back the gradient to the max values.
    for (size_t i = 0U; i < input_size; ++i)
    {
        for (size_t j = 0U; j < input_size; ++j)
        {
            input_gradients[i * input_size + j] = gradients[i * input_size + j];
        }
    }
    // Return true to indicate success.
    return true;
}
