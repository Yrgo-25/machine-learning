/**
 * @file Machine learning utility functions.
 */
#pragma once

#include <cstddef>

#include "ml/types.h"

namespace ml
{
/**
 * @brief Initialize one-dimensional matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix1d& matrix) noexcept;

/**
 * @brief Initialize two-dimensional matrix with zeros.
 *
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix2d& matrix) noexcept;

/**
 * @brief Initialize one-dimensional matrix with zeros, resizing it if necessary.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix1d& matrix, std::size_t size) noexcept;

/**
 * @brief Initialize two-dimensional matrix with zeros, resizing it if necessary.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix2d& matrix, std::size_t size) noexcept;

/**
 * @brief Initialize two-dimensional matrix with zeros, resizing it if necessary.
 *
 * @param[in] matrix The matrix to initialize.
 * @param[in] rowCount The desired row count of the matrix.
 * @param[in] colCount The desired column count of the matrix.
 */
void initMatrix(Matrix2d& matrix, std::size_t rowCount, std::size_t colCount) noexcept;

/**
 * @brief Check whether given matrix is square.
 *
 * @param[in] matrix The matrix to check.
 *
 * @return True if given matrix is square, false otherwise.
 */
[[nodiscard]] bool isMatrixSquare(const Matrix2d& matrix) noexcept;

/**
 * @brief Check whether the given row in given matrix is square.
 *
 * @param[in] matrix The matrix to check.
 * @param[in] row The row to check.
 *
 * @return True if given row is square, false otherwise.
 */
[[nodiscard]] bool isMatrixSquare(const Matrix2d& matrix, std::size_t row) noexcept;

/**
 * @brief Print the contents of given one-dimensional matrix.
 *
 * @param[in] matrix The matrix to print.
 */
void printMatrix(const Matrix1d& matrix) noexcept;

/**
 * @brief Print the contents of given two-dimensional matrix.
 *
 * @param[in] matrix The matrix to print.
 */
void printMatrix(const Matrix2d& matrix) noexcept;

/**
 * @brief Match dimensions.
 *
 * @param[in] expectedSize Expected size.
 * @param[in] actualSize Actual size.
 *
 * @return True if the dimensions match, false otherwise.
 */
[[nodiscard]] bool matchDimensions(std::size_t expectedSize, std::size_t actualSize) noexcept;

/**
 * @brief Check learning rate.
 *
 * @param[in] learningRate The learning rate to check.
 *
 * @return True if the learning rate is valid, false otherwise.
 */
[[nodiscard]] bool checkLearningRate(double learningRate) noexcept;

/**
 * @brief Get a randomized starting value for trainable parameters.
 *
 * @return Random value in the range [0.0, 1.0] (inclusive).
 */
[[nodiscard]] double randomStartVal() noexcept;

/**
 * @brief Create a training order list.
 *
 * @param[in] trainSetCount The number of training sets.
 *
 * @return The new training order list.
 */
[[nodiscard]] TrainOrderList createTrainOrderList(std::size_t trainSetCount) noexcept;

/**
 * @brief Shuffle training order list.
 *
 * @param[in] list Training order list to shuffle.
 */
void shuffleTrainOrderList(TrainOrderList& list) noexcept;

} // namespace ml
