//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_POLYNOMIAL_H
#define POLYNOMIAL_POLYNOMIAL_H

#include <stdio.h>
#include <stdbool.h>

#include "compat_complex.h"

typedef struct polynomial polynomial_t;

#define MAX_BATCH_SIZE   40960
#define MAX_BATCH_MEM_MB 100

struct polynomial {
  size_t degree;

  // size of arrays all the same as degree
  // except coeffs which is deg+1, but may not be filled
  // if for example there are repeated roots
  cxldouble *coeffs;   // ordered with k(N)x^N + K(N-1)x^(N-1) ... + k(1)x + k(0)
  cxldouble *roots;    // distinct roots in no particular ordering
  size_t *multiplicity;     // multiplicity of each root
  size_t num_distinct_roots;

  bool roots_valid;
  bool coeffs_valid;
};

// initializes a default polynomial with roots and coefficients uninitialized
polynomial_t *polynomial_new(size_t degree);

// create a polynomial with specified roots in no particular order
polynomial_t *polynomial_from_roots(cxldouble *roots, size_t num_roots, bool dedup);
polynomial_t *polynomial_from_dis_roots(cxldouble *distinct_roots, size_t *mult, size_t num_distinct_roots);

// create a polynomial with coefficients
// ordered with k(N)x^N + K(N-1)x^(N-1) ... + k(1)x + k(0)
polynomial_t *polynomial_from_coeffs(cxldouble *coeffs, size_t num_coeffs);

// free memory from polynomial struct
void polynomial_free(polynomial_t *poly);

// allocates and returns a new polynomial
polynomial_t *polynomial_copy(polynomial_t *poly);

// evaluate polynomial at point z
cxldouble polynomial_eval(polynomial_t *poly, cxldouble z);

// find all roots of polynomial, stores them in poly->roots
// returns true on success
bool polynomial_find_roots(polynomial_t *poly);

// find roots for all coefficient combinations
// given N base coefficients and degree M, generates N^(M+1) polynomials
// by substituting each base coeff into each position
// roots array should be num_combinations * poly_degree
// num_distinct array should be num_combinations
// returns total roots found
size_t polynomial_find_root_combinations(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations);

size_t polynomial_find_root_combinations_companion(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations);

// only computes every skip-th combination
// skipped combinations have num_distinct[i] = 0 to maintain hue index
size_t polynomial_find_root_combinations_skip(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip);

size_t polynomial_find_root_combinations_companion_skip(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip);

#endif //POLYNOMIAL_POLYNOMIAL_H