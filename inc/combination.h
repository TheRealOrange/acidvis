//
// Created by Lin Yicheng on 5/12/25.
//

#ifndef POLYNOMIAL_COMBINATION_H
#define POLYNOMIAL_COMBINATION_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// find roots for all coefficient combinations
// given N base coefficients and degree M, generates N^(M+1) polynomials
// by substituting each base coeff into each position
// roots array should be num_combinations * poly_degree
// num_distinct array should be num_combinations
// returns total roots found
// only computes every skip-th combination
// skipped combinations have num_distinct[i] = 0 to maintain hue index
size_t polynomial_find_root_combinations(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip, bool use_lapack);

#endif //POLYNOMIAL_COMBINATION_H