/**
 * @brief Convolutional layer implementation details.
 */
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "ml/conv_layer.h"
#include "ml/matrix.h"

/**
 * @brief Convolutional layer structure.
 */
typedef struct conv_layer
{
    /** Input data with padding. */
    matrix_t* input_padded;

    /** Input gradients with padding. */
    matrix_t* input_gradients_padded;

    /** Input gradients (unpadded). */
    matrix_t* input_gradients;

    /** Kernel matrix (weights). */
    matrix_t* kernel;

    /** Kernel gradients. */
    matrix_t* kernel_gradients;

    /** Output data. */
    matrix_t* output;

    /** Pre-activation output. */
    matrix_t* preactivation_output;

    /** Bias value. */
    double bias;

    /** Bias gradients. */
    double bias_gradient;
} conv_layer_t;

// -----------------------------------------------------------------------------
static void init_rand(void)
{
    static bool initialized = false;

    if (!initialized)
    {
        srand(time(NULL));
        initialized = true;
    }
}

// -----------------------------------------------------------------------------
static inline double rand_start_val() { return (double)(rand()) / RAND_MAX; }

// -----------------------------------------------------------------------------
static inline double relu_output(const double input) { return 0.0 < input ? input : 0.0; }

// -----------------------------------------------------------------------------
static inline double relu_delta(const double input) { return 0.0 < input ? 1.0 : 0.0; }

// -----------------------------------------------------------------------------
static size_t conv_layer_padded_size(const conv_layer_t* self)
{
    return sqrt(matrix_size(self->input_padded));
}

// -----------------------------------------------------------------------------
static size_t conv_layer_kernel_size(const conv_layer_t* self)
{
    return sqrt(matrix_size(self->kernel));
}

// -----------------------------------------------------------------------------
static size_t conv_layer_output_size(const conv_layer_t* self)
{
    return sqrt(matrix_size(self->output));
}

// -----------------------------------------------------------------------------
static size_t conv_layer_pad_offset(const conv_layer_t* self)
{
    return conv_layer_kernel_size(self) / 2U;
}

// -----------------------------------------------------------------------------
static void conv_layer_pad(conv_layer_t* self, const matrix_t* input)
{
    // Fill the padded input matrix with zeros.
    matrix_init(self->input_padded);

    const size_t padded_size = conv_layer_padded_size(self);
    const size_t output_size = conv_layer_output_size(self);
    const size_t offset      = conv_layer_pad_offset(self);

    // Copy the input data to the padded matrix.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            const size_t padded_idx = (i + offset) * padded_size + (j + offset);
            const size_t input_idx  = i * output_size + j;

            double* input_padded     = matrix_data(self->input_padded);
            input_padded[padded_idx] = matrix_data_const(input)[input_idx];
        }
    }
}

// -----------------------------------------------------------------------------
static void conv_layer_extract_input_gradients(conv_layer_t* self)
{
    const size_t padded_size = conv_layer_padded_size(self);
    const size_t output_size = conv_layer_output_size(self);
    const size_t offset      = conv_layer_pad_offset(self);

    // Extract the input gradients from the corresponding padded vector.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            const size_t padded_idx = (i + offset) * padded_size + (j + offset);
            const size_t input_idx  = i * output_size + j;
            const double num        = matrix_data_const(self->input_gradients_padded)[padded_idx];

            matrix_data(self->input_gradients)[input_idx] = num;
        }
    }
}

// -----------------------------------------------------------------------------
conv_layer_t* conv_layer_new(const size_t input_size, const size_t kernel_size)
{
    if ((0U == kernel_size) || (0U == input_size) || (input_size < kernel_size)) { return NULL; }

    conv_layer_t* self = (conv_layer_t*)(malloc(sizeof(conv_layer_t)));
    if (NULL == self) { return NULL; }

    // Initialize the random generator (only done once) before drawing any random value.
    init_rand();

    const size_t pad_offset  = kernel_size / 2U;
    const size_t padded_size = input_size + 2U * pad_offset;

    self->input_padded           = matrix_new(padded_size * padded_size);
    self->input_gradients_padded = matrix_new(padded_size * padded_size);
    self->input_gradients        = matrix_new(input_size * input_size);
    self->kernel                 = matrix_new(kernel_size * kernel_size);
    self->kernel_gradients       = matrix_new(kernel_size * kernel_size);
    self->output                 = matrix_new(input_size * input_size);
    self->preactivation_output   = matrix_new(input_size * input_size);
    self->bias                   = rand_start_val();
    self->bias_gradient          = 0.0;

    if ((NULL == self->input_padded) || (NULL == self->input_gradients_padded) ||
        (NULL == self->input_gradients) || (NULL == self->kernel) ||
        (NULL == self->kernel_gradients) || (NULL == self->output) ||
        (NULL == self->preactivation_output))
    {
        conv_layer_del(&self);
        return NULL;
    }

    for (size_t ki = 0U; ki < kernel_size; ++ki)
    {
        for (size_t kj = 0U; kj < kernel_size; ++kj)
        {
            matrix_data(self->kernel)[ki * kernel_size + kj] = rand_start_val();
        }
    }
    return self;
}

// -----------------------------------------------------------------------------
void conv_layer_del(conv_layer_t** self)
{
    if ((NULL == self) || (NULL == *self)) { return; }

    // Free allocated resources and set the associated pointer to null.
    conv_layer_t* impl = *self;
    matrix_del(&(impl->input_padded));
    matrix_del(&(impl->input_gradients_padded));
    matrix_del(&(impl->input_gradients));
    matrix_del(&(impl->kernel));
    matrix_del(&(impl->kernel_gradients));
    matrix_del(&(impl->output));
    matrix_del(&(impl->preactivation_output));
    free(impl);
    *self = NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* conv_layer_output(const conv_layer_t* self)
{
    return NULL != self ? self->output : NULL;
}

// -----------------------------------------------------------------------------
const matrix_t* conv_layer_input_gradients(const conv_layer_t* self)
{
    return NULL != self ? self->input_gradients : NULL;
}

// -----------------------------------------------------------------------------
bool conv_layer_feedforward(conv_layer_t* self, const matrix_t* input)
{
    if ((NULL == self) || (NULL == input)) { return false; }
    if (matrix_size(input) != matrix_size(self->output)) { return false; }

    const size_t padded_size = conv_layer_padded_size(self);
    const size_t output_size = conv_layer_output_size(self);
    const size_t kernel_size = conv_layer_kernel_size(self);

    conv_layer_pad(self, input);

    // Perform convolution and activation.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            // Compute the sum for the given region.
            double sum = self->bias;

            for (size_t ki = 0U; ki < kernel_size; ++ki)
            {
                for (size_t kj = 0U; kj < kernel_size; ++kj)
                {
                    const size_t input_idx  = (i + ki) * padded_size + (j + kj);
                    const size_t kernel_idx = ki * kernel_size + kj;

                    const double input_padded = matrix_data_const(self->input_padded)[input_idx];
                    const double kernel       = matrix_data_const(self->kernel)[kernel_idx];
                    sum += input_padded * kernel;
                }
            }
            // Apply activation function over the calculated sum.
            const size_t output_index                             = i * output_size + j;
            matrix_data(self->preactivation_output)[output_index] = sum;
            matrix_data(self->output)[output_index]               = relu_output(sum);
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool conv_layer_backpropagate(conv_layer_t* self, const matrix_t* output_gradients)
{
    if ((NULL == self) || (NULL == output_gradients)) { return false; }
    if (matrix_size(output_gradients) != matrix_size(self->output)) { return false; }

    matrix_init(self->input_gradients_padded);
    matrix_init(self->input_gradients);
    matrix_init(self->kernel_gradients);
    self->bias_gradient = 0.0;

    const size_t padded_size = conv_layer_padded_size(self);
    const size_t output_size = conv_layer_output_size(self);
    const size_t kernel_size = conv_layer_kernel_size(self);

    // Compute gradients for all parameters.
    for (size_t i = 0U; i < output_size; ++i)
    {
        for (size_t j = 0U; j < output_size; ++j)
        {
            // Compute local gradient (delta) using activation derivative.
            const size_t out_idx  = i * output_size + j;
            const double out      = matrix_data_const(self->preactivation_output)[out_idx];
            const double out_grad = matrix_data_const(output_gradients)[out_idx];
            const double delta    = out_grad * relu_delta(out);
            self->bias_gradient += delta;

            // Accumulate gradients for kernel and input.
            for (size_t ki = 0U; ki < kernel_size; ++ki)
            {
                for (size_t kj = 0U; kj < kernel_size; ++kj)
                {
                    const size_t input_idx  = (i + ki) * padded_size + (j + kj);
                    const size_t kernel_idx = (ki * kernel_size) + kj;
                    const double input      = matrix_data_const(self->input_padded)[input_idx];
                    const double kernel     = matrix_data_const(self->kernel)[kernel_idx];

                    matrix_data(self->input_gradients_padded)[input_idx] += kernel * delta;
                    matrix_data(self->kernel_gradients)[kernel_idx] += input * delta;
                }
            }
        }
    }
    conv_layer_extract_input_gradients(self);
    return true;
}

// -----------------------------------------------------------------------------
bool conv_layer_optimize(conv_layer_t* self, const double learning_rate)
{
    if ((NULL == self) || (0.0 >= learning_rate)) { return false; }

    self->bias -= self->bias_gradient * learning_rate;

    const size_t kernel_size = conv_layer_kernel_size(self);

    for (size_t ki = 0U; ki < kernel_size; ++ki)
    {
        for (size_t kj = 0U; kj < kernel_size; ++kj)
        {
            const size_t index           = ki * kernel_size + kj;
            const double kernel_gradient = matrix_data_const(self->kernel_gradients)[index];

            matrix_data(self->kernel)[ki * kernel_size + kj] -= kernel_gradient * learning_rate;
        }
    }
    return true;
}
