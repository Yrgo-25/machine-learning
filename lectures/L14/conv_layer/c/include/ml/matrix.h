/**
 * @file Matrix implementations for machine learning.
 */
#ifndef ML_MATRIX_H_
#define ML_MATRIX_H_

#include <stdbool.h>
#include <stddef.h>

/** Matrix structure. */
typedef struct matrix matrix_t;

/**
 * @brief Create a new matrix.
 *
 * @param[in] size Size of the matrix.
 *
 * @return Pointer to the new matrix, or a nullptr on failure.
 */
matrix_t* matrix_new(size_t size);

/**
 * @brief Delete given matrix.
 *
 *        Release allocated resources and set the associated pointer to null.
 *
 * @param[in] self Double pointer to the matrix to delete.
 */
void matrix_del(matrix_t** self);

/**
 * @brief Get matrix size.
 *
 * @param[in] self Pointer to the matrix.
 *
 * @return The size of the matrix.
 *
 * @note Ensure that the matrix is valid before invoking this function.
 */
size_t matrix_size(const matrix_t* self);

/**
 * @brief Get matrix data.
 *
 * @param[in] self Pointer to the matrix.
 *
 * @return Pointer to the matrix data.
 *
 * @note Ensure that the matrix is valid before invoking this function.
 */
double* matrix_data(matrix_t* self);

/**
 * @brief Get matrix data (read-only).
 *
 * @param[in] self Pointer to the matrix.
 *
 * @return Pointer to the matrix data.
 *
 * @note Ensure that the matrix is valid before invoking this function.
 */
const double* matrix_data_const(const matrix_t* self);

/**
 * @brief Check whether given matrix is empty.
 *
 * @param[in] self Pointer to the matrix.
 *
 * @return True if the matrix is empty, false otherwise.
 */
bool matrix_empty(const matrix_t* self);

/**
 * @brief Initialize given matrix with zeros.
 *
 * @param[in] self Matrix to initialize.
 */
void matrix_init(matrix_t* self);

/**
 * @brief Create a new matrix by copying given data.
 *
 * @param[in] buffer Buffer holding the data to copy.
 * @param[in] bytes Size of the buffer in bytes.
 *
 * @return Pointer to the new matrix, or a nullptr on failure.
 */
matrix_t* matrix_copy(const void* buffer, size_t bytes);

/**
 * @brief Print the contents of given matrix to the terminal.
 *
 * @param[in] self The matrix to print.
 * @param[in] print2d True to print the matrix as a 2D matrix.
 */
void matrix_print(const matrix_t* self, bool print2d);

#endif /** ML_MATRIX_H_ */
