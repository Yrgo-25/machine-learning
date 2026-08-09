/**
 * @file Simple flatten layer demo.
 */
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/**
 * @brief Initialize matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix1d& matrix) noexcept
{
    // Fill the matrix with zeros.
    for (auto& num : matrix)
    {
        num = 0.0;
    }
}

/**
 * @brief Initialize matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix2d& matrix) noexcept
{
    // Fill the matrix with zeros.
    for (auto& row : matrix)
    {
        for (auto& num : row)
        {
            num = 0.0;
        }
    }
}

/**
 * @brief Initialize matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix1d& matrix, const std::size_t size)
{
    // Resize the matrix if necessary, then fill the matrix with zeros.
    matrix.resize(size);
    initMatrix(matrix);
}

/**
 * @brief Initialize matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix2d& matrix, const std::size_t size)
{
    // Resize the matrix if necessary, then fill the matrix with zeros.
    matrix.resize(size, Matrix1d(size));
    initMatrix(matrix);
}

/**
 * @brief Check whether given matrix is square.
 *
 * @param[in] matrix The matrix to check.
 *
 * @return True if the matrix is square, false otherwise.
 */
[[nodiscard]] bool isMatrixSquare(const Matrix2d& matrix) noexcept
{
    // Check the matrix row by row.
    for (const auto& row : matrix)
    {
        // Return false if the matrix isn't square.
        if (row.size() != matrix.size()) { return false; }
    }
    // Return true to indicate that the matrix is square.
    return true;
}

/**
 * @brief Print the contents of given matrix.
 *
 * @param[in] matrix The matrix to print.
 */
void printMatrix(const Matrix1d& matrix) noexcept
{
    const auto* lastNum{matrix.empty() ? nullptr : &matrix[matrix.size() - 1U]};

    // Print the matrix contents on a single row.
    for (const auto& num : matrix)
    {
        // Separate each number in the row with a comma.
        std::printf("%.1f", num);
        if (&num < lastNum) { std::printf(", "); }
    }
    std::printf("\n");
}

/**
 * @brief Print the contents of given matrix.
 *
 * @param[in] matrix The matrix to print.
 */
void printMatrix(const Matrix2d& matrix) noexcept
{
    // Print the matrix contents row by row.
    for (const auto& row : matrix)
    {
        const auto* lastNum{row.empty() ? nullptr : &row[row.size() - 1U]};
        std::printf("\t");

        // Separate each number in the row with a comma.
        for (const auto& num : row)
        {
            std::printf("%.1f", num);
            if (&num < lastNum) { std::printf(", "); }
        }
        std::printf("\n");
    }
    std::printf("\n");
}

namespace ml
{
/**
 * @brief Flatten layer structure.
 *
 * @todo Implement this struct!
 */
struct FlattenLayer final
{
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     */

    /**
     * @todo Check the input size in the constructor:
     *
     *       If the input size is 0, terminate the program with error message
     *       "Cannot create flatten layer: invalid input size!".
     */

    /**
     * @brief Flatten the input from 2D to 1D.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */

    /**
     * @brief Unflatten the output gradients from 1D to 2D.
     *
     * @param[in] outputGradients Matrix holding output gradients.
     *
     * @return True on success, false on failure.
     */

    /** @todo Define member variables here! Keep them public and skip the 'my' prefix. */
};
} // namespace ml
} // namespace

/**
 * @brief Create and demonstrate a simple flatten layer.
 *
 * @return 0 on success, -1 on failure.
 */
int main()
{
//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#ifdef FLATTEN_LAYER_IMPLEMENTED
    // Example 4x4 input matrix (could represent an image or feature map).
    const Matrix2d input{{2, 1, 6, 1}, {3, 0, 4, 6}, {1, 2, 4, 5}, {3, 4, 7, 7}};

    // Example output gradients (same shape as flattened output, used for backpropagation demo).
    const Matrix1d outputGradients{1, 2, 3, 4, 8, 7, 6, 5, 0, 2, 4, 8, 9, 7, 5, 3};

    // Create a flatten layer: 4x4 input, produces 1x16 output.
    constexpr std::size_t inputSize{4U};
    ml::FlattenLayer flattenLayer{inputSize};

    // Perform feedforward (flatten the input), print the result.
    std::printf("Flattening input data (2D -> 1D):\n");
    printMatrix(input);
    flattenLayer.feedforward(input);
    std::printf("Resulting flattened output (1D):\n");
    printMatrix(flattenLayer.output);

    // Perform backpropagation (unflatten the output), print the result.
    std::printf("\nApplying backpropagation (1D -> 2D):\n");
    printMatrix(outputGradients);
    flattenLayer.backpropagate(outputGradients);
    std::printf("\nResulting unflattened input gradients (2D):\n");
    printMatrix(flattenLayer.inputGradients);
    return 0;

//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** FLATTEN_LAYER_IMPLEMENTED */
}