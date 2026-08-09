"""Simple max pooling layer demo."""

# pylint: disable = consider-using-enumerate

# One-dimensional matrix.
Matrix1d = list[float]

# Two-dimensional matrix.
Matrix2d = list[Matrix1d]


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


def init_matrix2d(matrix: Matrix2d) -> None:
    """Initialize matrix with zeros.

    Args:
        matrix: The matrix to initialize.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i][j] = 0.0


def is_matrix2d_square(matrix: Matrix2d) -> bool:
    """Check whether matrix is square.

    Args:
        matrix: The matrix to check.

    Returns:
        True if the matrix is square, false otherwise.
    """
    return all(len(row) == len(matrix) for row in matrix)


class MaxPoolLayer:
    """Max pooling layer implementation."""

    def __init__(self, input_size: int, pool_size: int) -> None:
        # Check the input parameters, throw an exception if invalid.
        if input_size == 0 or pool_size == 0 or input_size % pool_size != 0:
            raise ValueError(
                "Cannot create max pooling layer: invalid input arguments!"
            )

        # Calculate the output size.
        output_size = input_size // pool_size

        # Initialize the member variables.
        self.input = create_matrix2d(input_size)
        self.input_gradients = create_matrix2d(input_size)
        self.output = create_matrix2d(output_size)

    def feedforward(self, input_data: Matrix2d) -> bool:
        """Perform feedforward operation.

        Args:
            input_data: Input data.

        Returns:
            True on success, false on failure.
        """
        # Check the input dimensions, return false on failure.
        if len(input_data) != len(self.input) or not is_matrix2d_square(input_data):
            return False
        # Compute the pool size.
        pool_size = len(self.input) // len(self.output)

        # Perform feedforward - extract the max value of each region.
        for i in range(len(self.output)):
            for j in range(len(self.output)):
                row = i * pool_size
                col = j * pool_size
                # Find the max value of each region.
                max_val = input_data[row][col]
                for pi in range(pool_size):
                    for pj in range(pool_size):
                        val = input_data[pi + row][pj + col]
                        if val > max_val:
                            max_val = val
                self.output[i][j] = max_val

        # Save the input for future backpropagation, then return true to indicate success.
        self.input = [row[:] for row in input_data]
        return True

    def backpropagate(self, output_gradients: Matrix2d) -> bool:
        """Perform backpropagation.

        Args:
            output_gradients: Output gradients from the next layer.

        Returns:
            True on success, false on failure.
        """
        # Check the output dimensions, return false on failure.
        if len(output_gradients) != len(self.output):
            return False
        # Check if the matrix is square, return false if not.
        if not all(len(row) == len(output_gradients) for row in output_gradients):
            return False

        # Compute the pool size.
        pool_size = len(self.input) // len(self.output)

        # Reinitialize the input gradients.
        init_matrix2d(self.input_gradients)

        # Perform backpropagation - feed back the gradient to the max values.
        for i in range(len(self.output)):
            for j in range(len(self.output)):
                row = i * pool_size
                col = j * pool_size

                max_val = self.output[i][j]
                max_row = row
                max_col = col

                # Find the first position of the max value in the input.
                found = False
                for pi in range(pool_size):
                    for pj in range(pool_size):
                        val = self.input[row + pi][col + pj]
                        if not found and val == max_val:
                            max_row = row + pi
                            max_col = col + pj
                            found = True
                    if found:
                        break
                # Feed back the gradient to the max position only.
                self.input_gradients[max_row][max_col] = output_gradients[i][j]
        # Return true to indicate success.
        return True


def main() -> None:
    """Create and demonstrate a simple max pooling layer."""

    # Example 4x4 input matrix (could represent an image or feature map).
    input_data = [
        [2, 1, 6, 1],
        [3, 0, 4, 6],
        [1, 2, 4, 5],
        [3, 4, 7, 7],
    ]

    # Example output gradients (same shape as pooling output, used for backpropagation demo).
    output_gradients = [
        [1, 2],
        [3, 4],
    ]

    # Create a max pooling layer: 4x4 input, 2x2 pooling regions, produces 2x2 output.
    pool_layer = MaxPoolLayer(4, 2)

    # Show the input matrix.
    print("Pooling input data (2D):")
    for row in input_data:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    # Perform feedforward (pooling).
    pool_layer.feedforward(input_data)
    print("\nPooled output (2D):")
    for row in pool_layer.output:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    # Show the output gradients.
    print("\nPooling output gradients (2D):")
    for row in output_gradients:
        print("  ", " ".join(f"{num:.1f}" for num in row))

    # Perform backpropagation.
    pool_layer.backpropagate(output_gradients)
    print("\nInput gradients after backpropagation (2D):")
    for row in pool_layer.input_gradients:
        print("  ", " ".join(f"{num:.1f}" for num in row))


# Invoke the main function if this is the startup script.
if __name__ == "__main__":
    main()
