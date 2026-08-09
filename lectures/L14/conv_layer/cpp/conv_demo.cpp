/**
 * @file Simple convolutional layer demo.
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace
{
/** One-dimensional matrix. */
using Matrix1d = std::vector<double>;

/** Two-dimensional matrix. */
using Matrix2d = std::vector<Matrix1d>;

/**
 * @brief Initialize given matrix with zeros.
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

/**
 * @brief Generate a random starting value between 0.0 and 1.0.
 *
 * @return Random floating-point value in the range [0.0, 1.0].
 */
double randomStartVal() noexcept
{
    constexpr auto max = static_cast<double>(RAND_MAX);
    return std::rand() / max;
}

/**
 * @brief ReLU activation function (output).
 *
 * @param[in] input Input value.
 *
 * @return Output after ReLU activation.
 */
constexpr double reluOutput(const double input) noexcept { return 0.0 < input ? input : 0.0; }

/**
 * @brief ReLU activation function (derivative).
 *
 * @param[in] input Input value.
 *
 * @return Derivative of ReLU at input.
 */
constexpr double reluDelta(const double input) noexcept { return 0.0 < input ? 1.0 : 0.0; }

namespace ml
{
/**
 * @brief Convolutional layer structure.
 *
 * @todo Implement this struct!
 */
struct ConvLayer final
{
    /**
     * @brief Constructor.
     *
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     */

    /**
     * @todo Check the input arguments in the constructor:
     *
     *       If the input size is 0, the kernel size is 0, or the input size is smaller than
     *       the kernel size, terminate the program with error message
     *       "Cannot create convolutional layer: invalid input arguments!".
     */

    /**
     * @brief Perform feedforward operation.
     *
     * @param[in] input Matrix holding input data.
     *
     * @return True on success, false on failure.
     */

    /**
     * @brief Perform backpropagation.
     *
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     *
     * @return True on success, false on failure.
     */

    /**
     * @brief Perform optimization.
     *
     * @param[in] learningRate Learning rate to use. Must be in range (0.0, 1.0].
     *
     * @return True on success, false on failure.
     */

    /** @todo Define member variables here! Keep them public and skip the 'my' prefix. */

private:
    /**
     * @brief Pad input with zeros.
     *
     * @param[in] input Input data.
     */

    /**
     * @brief Extract input gradients.
     */
};
} // namespace ml
} // namespace

/**
 * @brief Create and demonstrate a simple convolutional layer.
 *
 * @return 0 on success, -1 on failure.
 */
int main()
{
//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#ifdef CONV_LAYER_IMPLEMENTED

    // Example 4x4 input matrix (could represent an image or feature map).
    const Matrix2d input{{1, 1, 1, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 1, 1}};

    // Example output gradients (target output for demonstration).
    const Matrix2d outputGradients{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};

    // Initialize the random generator with the current time as seed.
    std::srand(std::time(nullptr));

    // Create a convolutional layer: 4x4 input, 2x2 kernel.
    constexpr std::size_t inputSize{4U};
    constexpr std::size_t kernelSize{2U};
    ml::ConvLayer convLayer{inputSize, kernelSize};

    // Show the input matrix.
    std::printf("Convolution input data (2D):\n");
    printMatrix(input);

    // Perform feedforward (convolution).
    convLayer.feedforward(input);
    std::printf("Convolution output (2D):\n");
    printMatrix(convLayer.output);

    // Show the output gradients.
    std::printf("Convolution output gradients (2D):\n");
    printMatrix(outputGradients);

    // Perform backpropagation.
    convLayer.backpropagate(outputGradients);
    std::printf("Input gradients after backpropagation (2D):\n");
    printMatrix(convLayer.inputGradients);
    return 0;

//! @todo Remove this guard (and/or uncomment the compiler flags in the makefile) once the
//        implementation is finished.
#endif /** CONV_LAYER_IMPLEMENTED */
}
