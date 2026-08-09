/**
 * @brief Max pooling layer implementation details.
 */
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "ml/matrix.h"
#include "ml/max_pool_layer.h"

/**
 * @brief Max pooling layer structure.
 */
typedef struct max_pool_layer
{
    /** Input data. */
    matrix_t* input;

    /** Input gradients. */
    matrix_t* input_gradients;

    /** Output data. */
    matrix_t* output;
} max_pool_layer_t;

// -----------------------------------------------------------------------------
static size_t max_pool_layer_input_size(const max_pool_layer_t* self)
{
    return sqrt(matrix_size(self->input));
}

// -----------------------------------------------------------------------------
static size_t max_pool_layer_output_size(const max_pool_layer_t* self)
{
    return sqrt(matrix_size(self->output));
}

// -----------------------------------------------------------------------------
static size_t max_pool_layer_pool_size(const max_pool_layer_t* self)
{
    return max_pool_layer_input_size(self) / max_pool_layer_output_size(self);
}

// -----------------------------------------------------------------------------
static void max_pool_layer_copy_input(max_pool_layer_t* self, const matrix_t* input)
{
    const size_t input_size = max_pool_layer_input_size(self);
    const double* source    = matrix_data_const(input);
    double* copy            = matrix_data(self->input);

    for (size_t i = 0U; i < input_size; ++i)
    {
        for (size_t j = 0U; j < input_size; ++j)
        {
            copy[i * input_size + j] = source[i * input_size + j];
        }
    }
}

// -----------------------------------------------------------------------------
max_pool_layer_t* max_pool_layer_new(const size_t input_size, const size_t pool_size)
{
    // Check the input parameters, return null if invalid.
    if ((0U == input_size) || (0U == pool_size)) { return NULL; }
    const bool dividible = (0U == (input_size % pool_size));
    if (!dividible) { return NULL; }

    // Create new max pooling layer, return null on failure.
    max_pool_layer_t* self = (max_pool_layer_t*)(malloc(sizeof(max_pool_layer_t)));
    if (NULL == self) { return NULL; }

    // Calculate output size.
    const size_t output_size = input_size / pool_size;

    // Initialize the member variables.
    self->input           = matrix_new(input_size * input_size);
    self->input_gradients = matrix_new(input_size * input_size);
    self->output          = matrix_new(output_size * output_size);

    // Check whether the member variables were initialize correctly, return null on failure.
    if ((NULL == self->input) || (NULL == self->input_gradients) || (NULL == self->output))
    {
        max_pool_layer_del(&self);
        return NULL;
    }
    // Return the new max pooling layer.
    return self;
}

// -----------------------------------------------------------------------------
void max_pool_layer_del(max_pool_layer_t** self)
{
    // Check if the layer is valid, terminate the function if not.
    if ((NULL == self) || (NULL == *self)) { return; }

    // Free allocated resources and set the associated pointer to null.
    max_pool_layer_t* impl = *self;
    matrix_del(&(impl->input));
    matrix_del(&(impl->input_gradients));
    matrix_del(&(impl->output));
    free(impl);
    *self = NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* max_pool_layer_output(const max_pool_layer_t* self)
{
    return NULL != self ? self->output : NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* max_pool_layer_input_gradients(const max_pool_layer_t* self)
{
    return NULL != self ? self->input_gradients : NULL;
}

// -----------------------------------------------------------------------------
bool max_pool_layer_feedforward(max_pool_layer_t* self, const matrix_t* input)
{
    // Check the input parameters, return false on failure.
    if ((NULL == self) || (NULL == input)) { return false; }
    if ((matrix_size(input) != matrix_size(self->input))) { return false; }

    const size_t input_size  = max_pool_layer_input_size(self);
    const size_t output_size = max_pool_layer_output_size(self);
    const size_t pool_size   = max_pool_layer_pool_size(self);

    const double* indata = matrix_data_const(input);
    double* output       = matrix_data(self->output);

    // Perform feedforward - extract the max value of each region.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            const size_t row = i * pool_size;
            const size_t col = j * pool_size;
            double max_val   = indata[row * input_size + col];

            // Find the max value of each region.
            for (size_t pi = 0U; pi < pool_size; ++pi)
            {
                for (size_t pj = 0U; pj < pool_size; ++pj)
                {
                    const size_t input_idx = (row + pi) * input_size + (col + pj);
                    const double val       = indata[input_idx];
                    if (val > max_val) { max_val = val; }
                }
            }
            output[i * output_size + j] = max_val;
        }
    }
    // Save the input for future backpropagation, then return true to indicate success.
    max_pool_layer_copy_input(self, input);
    return true;
}

// -----------------------------------------------------------------------------
bool max_pool_layer_backpropagate(max_pool_layer_t* self, const matrix_t* output_gradients)
{
    // Check the input parameters, return false on failure.
    if ((NULL == self) || (NULL == output_gradients)) { return false; }
    if (matrix_size(output_gradients) != matrix_size(self->output)) { return false; }

    // Reinitialize gradients with zeros.
    matrix_init(self->input_gradients);

    const size_t input_size  = max_pool_layer_input_size(self);
    const size_t output_size = max_pool_layer_output_size(self);
    const size_t pool_size   = max_pool_layer_pool_size(self);

    const double* gradients = matrix_data_const(output_gradients);
    const double* input     = matrix_data_const(self->input);
    const double* output    = matrix_data_const(self->output);
    double* input_gradients = matrix_data(self->input_gradients);

    // Perform backpropagation - feed back the gradient to the max values.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            // Find the first position of the max value in the input.
            const double max_val = output[i * output_size + j];
            const size_t row     = i * pool_size;
            const size_t col     = j * pool_size;

            bool found     = false;
            size_t max_row = row;
            size_t max_col = col;

            for (size_t pi = 0U; pi < pool_size; ++pi)
            {
                for (size_t pj = 0U; pj < pool_size; ++pj)
                {
                    const size_t input_idx = (row + pi) * input_size + (col + pj);
                    const double val       = input[input_idx];

                    if (!found && (val == max_val))
                    {
                        max_row = row + pi;
                        max_col = col + pj;
                        found   = true;
                    }
                }
                if (found) { break; }
            }
            // Feed back the gradient to the max position only.
            input_gradients[max_row * input_size + max_col] = gradients[i * output_size + j];
        }
    }
    // Return true to indicate success.
    return true;
}
