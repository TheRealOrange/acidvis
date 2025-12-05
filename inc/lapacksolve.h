#ifndef POLYNOMIAL_LAPACKSOLVE_H
#define POLYNOMIAL_LAPACKSOLVE_H

#include "polynomial.h"
#include <stdbool.h>

#ifdef HAVE_LAPACK

// Find all roots of a polynomial using the companion matrix method
// The eigenvalues of the companion matrix are the roots of the polynomial
//
// Parameters:
//   poly       - Input polynomial (must have valid coefficients)
//   roots      - Output array for roots (must have space for poly->degree roots)
//   num_roots  - Output: number of roots found
//
// Returns: true on success, false on failure
//
// Note: This method converts to double precision internally for LAPACK compatibility.
//       Results are converted back to long double but some precision may be lost.
bool _polynomial_find_roots_companion(polynomial_t *poly,
                                     cxldouble *roots,
                                     size_t *num_roots);

// Find roots and store them directly in the polynomial structure
// optional deduplication and multiplicity
bool polynomial_find_roots_companion(polynomial_t *poly, bool dedup);

// batched eigenvalue computation for multiple companion matrices
// processes batch_size matrices in parallel with memory awareness
// matrices and eigenvalues must be contiguous arrays
// returns number of successful eigenvalue computations
size_t compute_eigenvalues_batch(
    cxdouble *matrices,      // input: batch_size × n × n matrices
    size_t n,                       // matrix dimension
    size_t batch_size,              // number of matrices
    cxdouble *eigenvalues);  // output: batch_size × n eigenvalues

#endif // HAVE_LAPACK

#endif // POLYNOMIAL_LAPACKSOLVE_H