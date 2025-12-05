//
// Created by Lin Yicheng on 5/12/25.
//

#ifndef POLYNOMIAL_COMBINATION_H
#define POLYNOMIAL_COMBINATION_H

#include "compat_complex.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// statistics about polynomial solving
typedef struct {
  size_t num_incremental;       // num solved incrementally
  size_t num_fullsolve;         // num needed full solve
  size_t num_incremental_failed; // num incremental that fall-backed to full
} solve_stats_t;

// find roots for all coefficient combinations (full solve, no caching)
// given N base coefficients and degree M, generates N^(M+1) polynomials
// by substituting each base coeff into each position
// roots array should be num_combinations * poly_degree
// combination_valid array should be num_combinations
// returns total roots found
// only computes every skip-th combination
// skipped combinations have combination_valid[i] = false to maintain hue index
// if stats is not NULL, populated with stats
size_t polynomial_find_root_combinations(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble *roots,
    bool *combination_valid,
    int *since_last_update,
    size_t num_combinations,
    size_t skip,
    bool use_lapack,
    solve_stats_t *stats);

// cached version for drag operations with incremental solving
// uses previous coefficients and roots to speed up computation
// since_last_update tracks when each combination was last fully solved
// if incremental is true, use incremental solve
// if stats is not NULL, populated with stats
size_t polynomial_find_root_combinations_cached(
    const cxldouble *base_coeffs,
    const cxldouble *prev_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble *roots,
    bool *combination_valid,
    int *since_last_update,
    size_t num_combinations,
    size_t skip,
    bool use_lapack,
    solve_stats_t *stats);

#endif //POLYNOMIAL_COMBINATION_H