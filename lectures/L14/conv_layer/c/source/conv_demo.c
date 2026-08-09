/**
 * @brief Simple convolutional layer demo.
 */
#include <stdio.h>

#include "ml/conv_layer.h"
#include "ml/matrix.h"

/** Convolutional layer input/output) size. */
#define CONV_SIZE 4U

/** Convolutional layer kernel size. */
#define KERNEL_SIZE 2U

int main(void)
{
    // Example 4x4 input matrix (could represent an image or feature map).
    const double input_data[CONV_SIZE][CONV_SIZE] = {
        {1, 1, 1, 1},
        {1, 0, 0, 1},
        {1, 0, 0, 1},
        {1, 1, 1, 1},
    };
    matrix_t* input = matrix_copy(input_data, sizeof(input_data));

    // Example output gradients (target output for demonstration).
    const double gradients[CONV_SIZE][CONV_SIZE] = {
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
    };
    matrix_t* output_gradients = matrix_copy(gradients, sizeof(gradients));

    // Create a convolutional layer: 4x4 input, 2x2 kernel, return -1 on failure.
    conv_layer_t* conv_layer = conv_layer_new(CONV_SIZE, KERNEL_SIZE);

    // Terminate the program with error code -1 if the layer couldn't be created.
    if (NULL == conv_layer)
    {
        printf("Failed to create convolutional layer, aborting program!\n");
        return -1;
    }

    // Show the input matrix.
    printf("Convolution input data (2D):\n");
    matrix_print(input, true);

    // Perform feedforward (convolution).
    conv_layer_feedforward(conv_layer, input);
    printf("Convolution output (2D):\n");
    matrix_print(conv_layer_output(conv_layer), true);

    // Show the output gradients.
    printf("Convolution output gradients (2D):\n");
    matrix_print(output_gradients, true);

    // Perform backpropagation.
    conv_layer_backpropagate(conv_layer, output_gradients);
    printf("Input gradients after backpropagation (2D):\n");
    matrix_print(conv_layer_input_gradients(conv_layer), true);

    // Release allocated resources, then return 0 to indicate success.
    matrix_del(&input);
    matrix_del(&output_gradients);
    conv_layer_del(&conv_layer);
    return 0;
}
