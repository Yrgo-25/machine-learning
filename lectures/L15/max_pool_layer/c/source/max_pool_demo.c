/**
 * @brief Simple max pooling layer demo.
 */
#include <stdio.h>

#include "ml/matrix.h"
#include "ml/max_pool_layer.h"

/** Pooling layer input size. */
#define INPUT_SIZE 4U

/** Pool size. */
#define POOL_SIZE 2U

int main(void)
{
    // Example 4x4 input matrix (could represent an image or feature map).
    const double input_data[INPUT_SIZE][INPUT_SIZE] = {
        {2, 1, 6, 1},
        {3, 0, 4, 6},
        {1, 2, 4, 5},
        {3, 4, 7, 7},
    };
    matrix_t* input = matrix_copy(input_data, sizeof(input_data));

    // Example output gradients (same shape as pooling output, used for backpropagation demo).
    const double gradients[POOL_SIZE][POOL_SIZE] = {
        {1, 2},
        {3, 4},
    };
    matrix_t* output_gradients = matrix_copy(gradients, sizeof(gradients));

    // Create a max pooling layer: 4x4 input, 2x2 pooling regions, produces 2x2 output.
    max_pool_layer_t* pool_layer = max_pool_layer_new(INPUT_SIZE, POOL_SIZE);

    // Terminate the program with error code -1 if the layer couldn't be created.
    if (NULL == pool_layer)
    {
        printf("Failed to create max pooling layer, aborting program!\n");
        return -1;
    }

    // Show the input matrix.
    printf("Pooling input data (2D):\n");
    matrix_print(input, true);

    // Perform feedforward (pooling).
    max_pool_layer_feedforward(pool_layer, input);
    printf("Pooled output (2D):\n");
    matrix_print(max_pool_layer_output(pool_layer), true);

    // Show the output gradients.
    printf("Pooling output gradients (2D):\n");
    matrix_print(output_gradients, true);

    // Perform backpropagation.
    max_pool_layer_backpropagate(pool_layer, output_gradients);
    printf("Input gradients after backpropagation (2D):\n");
    matrix_print(max_pool_layer_input_gradients(pool_layer), true);

    // Release allocated resources, then return 0 to indicate success.
    matrix_del(&input);
    matrix_del(&output_gradients);
    max_pool_layer_del(&pool_layer);
    return 0;
}
