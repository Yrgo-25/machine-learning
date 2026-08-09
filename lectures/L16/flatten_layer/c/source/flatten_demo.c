/**
 * @brief Simple flatten layer demo.
 */
#include <stdio.h>

#include "ml/flatten_layer.h"
#include "ml/matrix.h"

/** Flatten layer input size. */
#define INPUT_SIZE 4U

/** Flatten layer output size. */
#define OUTPUT_SIZE INPUT_SIZE* INPUT_SIZE

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

    // Example output gradients (same shape as flattened output, used for backpropagation demo).
    const double gradients[OUTPUT_SIZE] = {1, 2, 3, 4, 8, 7, 6, 5, 0, 2, 4, 8, 9, 7, 5, 3};
    matrix_t* output_gradients          = matrix_copy(gradients, sizeof(gradients));

    // Create a flatten layer: 4x4 input, produces 1x16 output.
    flatten_layer_t* flatten_layer = flatten_layer_new(INPUT_SIZE);

    // Terminate the program with error code -1 if the layer couldn't be created.
    if (NULL == flatten_layer)
    {
        printf("Failed to create flatten layer, aborting program!\n");
        return -1;
    }

    // Show the input matrix.
    printf("Flattening input data (2D -> 1D):\n");
    matrix_print(input, true);

    // Perform feedforward (flattening).
    flatten_layer_feedforward(flatten_layer, input);
    printf("Resulting flattened output (1D):\n");
    matrix_print(flatten_layer_output(flatten_layer), false);

    // Show the output gradients.
    printf("Applying backpropagation (1D -> 2D):\n");
    matrix_print(output_gradients, false);

    // Perform backpropagation.
    flatten_layer_backpropagate(flatten_layer, output_gradients);
    printf("Resulting unflattened input gradients (2D):\n");
    matrix_print(flatten_layer_input_gradients(flatten_layer), true);

    // Release allocated resources, then return 0 to indicate success.
    matrix_del(&input);
    matrix_del(&output_gradients);
    flatten_layer_del(&flatten_layer);
    return 0;
}
