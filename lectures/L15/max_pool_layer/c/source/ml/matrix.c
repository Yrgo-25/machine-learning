/**
 * @brief Matrix implementation details.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "ml/matrix.h"

/**
 * @brief Matrix structure.
 */
typedef struct matrix
{
    /** Matrix data. */
    double* data;

    /** Matrix size. */
    size_t size;
} matrix_t;

// -----------------------------------------------------------------------------
static void matrix_print1d(const matrix_t* self)
{
    const size_t n        = self->size;
    const size_t last_num = 0U == n ? 0U : n - 1U;

    // Print the matrix contents on a single row.
    for (size_t i = 0U; i < n; ++i)
    {
        const double num = self->data[i];

        // Separate each number in the row with a comma.
        if (i < last_num) { printf("%.1f, ", num); }
        else { printf("%.1f\n\n", num); }
    }
}

// -----------------------------------------------------------------------------
static void matrix_print2d(const matrix_t* self)
{
    const size_t n = sqrt(self->size);

    // Print the matrix contents row by row.
    for (size_t i = 0U; i < n; ++i)
    {
        const size_t last_num = 0U == n ? 0U : n - 1U;
        printf("\t");

        for (size_t j = 0U; j < n; ++j)
        {
            const double num = self->data[i * n + j];

            // Separate each number in the row with a comma.
            if (j < last_num) { printf("%.1f, ", num); }
            else { printf("%.1f\n", num); }
        }
    }
    printf("\n");
}

// -----------------------------------------------------------------------------
matrix_t* matrix_new(const size_t size)
{
    // Create the new matrix, return null on failure.
    matrix_t* self = (matrix_t*)(malloc(sizeof(matrix_t)));
    if (NULL == self) { return NULL; }

    // Initialize the matrix.
    self->data = NULL;
    self->size = 0U;

    // Resize the data field, return null on failure.
    self->data = (double*)(malloc(sizeof(double) * size));

    if (NULL == self->data)
    {
        free(self);
        return NULL;
    }

    // Initialize the matrix with zeros, then return it.
    self->size = size;
    matrix_init(self);
    return self;
}

// -----------------------------------------------------------------------------
void matrix_del(matrix_t** self)
{
    // Check if the matrix is valid, terminate the function if not.
    if ((NULL == self) || (NULL == *self)) { return; }

    // Free allocated resources and set the associated pointer to null.
    free((*self)->data);
    free(*self);
    *self = NULL;
}

// -----------------------------------------------------------------------------
bool matrix_empty(const matrix_t* self) { return NULL == self ? true : 0U == self->size; }

// -----------------------------------------------------------------------------
size_t matrix_size(const matrix_t* self) { return NULL == self ? 0U : self->size; }

// -----------------------------------------------------------------------------
double* matrix_data(matrix_t* self) { return NULL == self ? NULL : self->data; }

// -----------------------------------------------------------------------------
const double* matrix_data_const(const matrix_t* self) { return NULL == self ? NULL : self->data; }

// -----------------------------------------------------------------------------
void matrix_init(matrix_t* self)
{
    // Check the matrix, terminate the function on failure.
    if (NULL == self) { return; }

    // Fill the matrix with zeros.
    for (size_t i = 0U; i < self->size; ++i)
    {
        self->data[i] = 0.0;
    }
}

// -----------------------------------------------------------------------------
matrix_t* matrix_copy(const void* buffer, const size_t bytes)
{
    double* data      = (double*)(buffer);
    const size_t size = bytes / sizeof(double);

    // Check the buffer content, return null if invalid.
    if (NULL == data) { return NULL; }

    // Create the new matrix, return null on failure.
    matrix_t* self = matrix_new(size);
    if (NULL == self) { return NULL; }

    // Copy from the given buffer, then return the matrix.
    for (size_t i = 0; i < size; ++i)
    {
        self->data[i] = data[i];
    }
    return self;
}

// -----------------------------------------------------------------------------
void matrix_print(const matrix_t* self, const bool print2d)
{
    // Check the matrix, terminate the function on failure.
    if (NULL == self) { return; }

    // Print as 1D or 2D.
    if (print2d) { matrix_print2d(self); }
    else { matrix_print1d(self); }
}
