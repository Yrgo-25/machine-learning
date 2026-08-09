"""Simple flatten layer demo."""

# pylint: disable = consider-using-enumerate

# One-dimensional matrix.
Matrix1d = list[float]

# Two-dimensional matrix.
Matrix2d = list[Matrix1d]


def create_matrix1d(size: int) -> Matrix1d:
    """Create a one-dimensional matrix.

    Args:
        size: Size of the matrix.

    Returns:
        New matrix initialized with zeros.
    """
    return [0.0] * size


def create_matrix2d(size: int) -> Matrix2d:
    """Create a two-dimensional matrix.

    Args:
        size: Size of the matrix.

    Returns:
        New matrix initialized with zeros.
    """
    matrix: Matrix2d = []
    for _ in range(size):
        row = [0.0] * size
        matrix.append(row)
    return matrix


def is_matrix2d_square(matrix: Matrix2d) -> bool:
    """Check whether matrix is square.

    Args:
        matrix: The matrix to check.

    Returns:
        True if the matrix is square, false otherwise.
    """
    return all(len(row) == len(matrix) for row in matrix)


class FlattenLayer:
    """Flatten layer implementation."""

    def __init__(self, input_size: int) -> None:
        # Check the input size, throw an exception if invalid.
        if input_size == 0:
            raise ValueError("Cannot create flatten layer: invalid input size!")

        # Initialize the matrices.
        self.input_gradients = create_matrix2d(input_size)
        self.output = create_matrix1d(input_size * input_size)

    def feedforward(self, input_data: Matrix2d) -> bool:
        """Perform feedforward operation.

        Args:
            input_data: Input data to flatten (2D -> 1D).

        Returns:
            True on success, false on failure.
        """
        # Compute the input size of the flatten layer.
        input_size = len(self.input_gradients)
        # Check the input dimensions, return false on failure.
        if len(input_data) != input_size or not is_matrix2d_square(input_data):
            return False

        # Flatten the input data.
        for i in range(input_size):
            for j in range(input_size):
                index = i * input_size + j
                self.output[index] = input_data[i][j]
        # Return true to indicate success.
        return True

    def backpropagate(self, output_gradients: Matrix1d) -> bool:
        """Perform backpropagation.

        Args:
            output_gradients: Output gradients to unflatten (1D -> 2D).

        Returns:
            True on success, false on failure.
        """
        input_size = len(self.input_gradients)
        output_size = len(self.output)

        # Check the output dimensions, return false on failure.
        if len(output_gradients) != output_size:
            return False

        # Unflatten the output gradients.
        for i in range(input_size):
            for j in range(input_size):
                index = i * input_size + j
                self.input_gradients[i][j] = output_gradients[index]
        # Return true to indicate success.
        return True


def main() -> None:
    """Create and demonstrate a simple flatten layer."""

    # Example 4x4 input matrix (could represent an image or feature map).
    input_data = [
        [2, 1, 6, 1],
        [3, 0, 4, 6],
        [1, 2, 4, 5],
        [3, 4, 7, 7],
    ]

    # Example output gradients (same shape as flattened output, used for backpropagation demo).
    output_gradients = [1, 2, 3, 4, 8, 7, 6, 5, 0, 2, 4, 8, 9, 7, 5, 3]

    # Create a flatten layer: 4x4 input, produces 1x16 output.
    flatten_layer = FlattenLayer(4)

    # Perform feedforward (flatten the input), print the result.
    print("Flattening input data (2D -> 1D):")
    for row in input_data:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    flatten_layer.feedforward(input_data)
    print("\nResulting flattened output (1D):")
    print("  ", " ".join(f"{num:.1f}" for num in flatten_layer.output))

    # Perform backpropagation (unflatten the output gradients), print the result.
    print("\nApplying backpropagation (1D -> 2D):")
    print("  Output gradients:", " ".join(f"{num:.1f}" for num in output_gradients))

    flatten_layer.backpropagate(output_gradients)
    print("\nResulting unflattened input gradients (2D):")
    for row in flatten_layer.input_gradients:
        print("  ", " ".join(f"{num:.1f}" for num in row))


# Invoke the main function if this is the startup script.
if __name__ == "__main__":
    main()
