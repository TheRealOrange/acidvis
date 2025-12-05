//
// Created by Lin Yicheng on 5/12/25.
//

#include "compat_complex.h"
#include "combination.h"
#include "lapacksolve.h"
#include "util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// convert combination index to coefficient assignment
// treats index as base-N number where each digit picks which base coeff to use
static void index_to_combination(size_t index, size_t num_base_coeffs, size_t num_positions,
                                  const cxldouble *base_coeffs,
                                  cxldouble *out_coeffs) {
  for (size_t pos = 0; pos < num_positions; pos++) {
    size_t which_base = index % num_base_coeffs;
    out_coeffs[pos] = base_coeffs[which_base];
    index /= num_base_coeffs;
  }
}

#ifdef HAVE_LAPACK
// batched LAPACK eigenvalue computation
// processes polynomials in chunks
static size_t process_batch_lapack(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    size_t start_idx,
    size_t batch_size,
    cxldouble *roots_out,
    size_t *num_distinct_out,
    size_t skip) {

  size_t n = poly_degree;
  size_t total_roots = 0;

  // allocate batch arrays for this chunk
  for (size_t it = 0; it <= batch_size * skip; it++) {
    if (it % skip != 0) {
      num_distinct_out[it] = 0; // mark as invalid
    }
  }

  // allocate only for valid matrices
  cxdouble *matrices = malloc(batch_size * n * n * sizeof(cxdouble));
  cxdouble *eigenvalues = malloc(batch_size * n * sizeof(cxdouble));

  if (!matrices || !eigenvalues) {
    free(matrices);
    free(eigenvalues);
    return 0;
  }

  // build companion matrices in parallel (now safe - each thread has unique valid_i)
  int batch_i;  // msvc openmp requires signed int declared outside
  #pragma omp parallel for schedule(static)
  for (batch_i = 0; batch_i < (int)batch_size; batch_i++) {
    size_t global_i = start_idx + batch_i * skip;

    // build polynomial coefficients (heap allocation instead of VLA)
    cxldouble *coeffs = malloc((n + 1) * sizeof(cxldouble));
    if (!coeffs) continue;

    index_to_combination(global_i, num_base_coeffs, n + 1, base_coeffs, coeffs);

    // normalize to monic
    cxldouble lead = coeffs[n];
    for (size_t i = 0; i <= n; i++) {
      coeffs[i] = cxdivl(coeffs[i], lead);
    }

    // build companion matrix in column-major order at compacted position
    cxdouble *matrix = matrices + (batch_i * n * n);
    memset(matrix, 0, n * n * sizeof(cxdouble));

    // subdiagonal: 1s
    for (size_t i = 1; i < n; i++) {
      matrix[i + (i-1) * n] = CX(1.0, 0.0);
    }

    // last column: -c_i (negated normalized coefficients)
    for (size_t i = 0; i < n; i++) {
      matrix[i + (n-1) * n] = cxl_to_cx(cxnegl(coeffs[i]));
    }

    free(coeffs);
  }

  // compute eigenvalues only for matrices not skipped
  size_t successful = 0;
  if (batch_size > 0) {
    successful = compute_eigenvalues_batch(matrices, n, batch_size, eigenvalues);
    if (!successful) {
      fprintf(stderr, "failed to solve eigenvalue batch with degree=%lu, batch_size=%lu\n", n, batch_size);
    }
  }

  free(matrices);

  // convert results back to long double and store at correct global indices
  for (size_t it = 0; it < batch_size; it++) {
    // convert eigenvalues to long double
    cxldouble *roots = roots_out + (it * n * skip);
    cxdouble *evals = eigenvalues + (it * n);

    for (size_t j = 0; j < n; j++) {
      roots[j] = cx_to_cxl(evals[j]);
    }

    num_distinct_out[it * skip] = n;
    total_roots += n;
  }

  free(eigenvalues);

  return total_roots;
}
#endif // HAVE_LAPACK

// process single combination and find its roots
static size_t process_single_combination(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    size_t combination_index,
    cxldouble *roots_out,
    size_t max_roots,
    bool use_lapack) {

  size_t num_positions = poly_degree + 1;

  cxldouble *coeffs = malloc(num_positions * sizeof(cxldouble));
  if (!coeffs) return 0;

  index_to_combination(combination_index, num_base_coeffs, num_positions, base_coeffs, coeffs);

  polynomial_t *work = polynomial_from_coeffs(coeffs, num_positions);
  free(coeffs);

  if (!work) return 0;

#ifdef HAVE_LAPACK
  if (use_lapack) {
    polynomial_find_roots_companion(work);
  } else {
    polynomial_find_roots(work);
  }
#else
  (void)use_lapack;
  polynomial_find_roots(work);
#endif

  size_t num_distinct = work->num_distinct_roots;
  if (num_distinct > max_roots) {
    num_distinct = max_roots;
  }

  memcpy(roots_out, work->roots, num_distinct * sizeof(cxldouble));

  polynomial_free(work);
  return num_distinct;
}

// find roots for all N^(M+1) coefficient combinations
// skip parameter: if > 1, only compute every skip-th combination
// skipped combinations will have num_distinct[i] = 0 to maintain hue index
static size_t polynomial_find_root_combinations_helper(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    bool use_lapack,
    size_t skip) {

  if (!base_coeffs || num_base_coeffs == 0 || poly_degree == 0 || !*roots || !*num_distinct) {
    printf("invalid args\n");
    return 0;
  }

  if (skip == 0) skip = 1;

  memset(*num_distinct, 0, num_combinations * sizeof(size_t));

  size_t total_roots = 0;

#ifdef HAVE_LAPACK
  // use batched processing for LAPACK if it's worth it
  if (use_lapack && num_combinations > 32) {
    size_t n = poly_degree;

    // calculate the number of polynomials we have
    // compute accounting for the skipped ones
    size_t num_polys = num_combinations / skip;
    size_t batch_size = MIN(MAX_BATCH_SIZE, num_polys);

    // process in chunks
    for (size_t start = 0; start < num_polys; start += batch_size) {
      size_t current_batch_size = batch_size;
      if (start + batch_size > num_polys) {
        current_batch_size = num_polys - start;
      }

      size_t start_idx = start * skip;

      // get pointers to this chunk's output
      cxldouble *batch_roots = *roots + (start_idx * poly_degree);
      size_t *batch_num_distinct = *num_distinct + start_idx;

      size_t batch_total = process_batch_lapack(
        base_coeffs, num_base_coeffs, poly_degree,
        start_idx, current_batch_size,
        batch_roots, batch_num_distinct, skip
      );

      total_roots += batch_total;
    }

    return total_roots;
  }
#endif

  // fallback, process one-by-one, for non-LAPACK or small batches
#ifdef _OPENMP
  int i;  // msvc openmp requires signed int declared outside
  #pragma omp parallel reduction(+:total_roots)
  {
    #pragma omp for schedule(dynamic, 64)
    for (i = 0; i < (int)num_combinations; i++) {
      // skip combinations if not divisible by skip value
      if (i % skip != 0) {
        // set num_distinct to 0 to maintain hue index
        (*num_distinct)[i] = 0;
        continue;
      }

      cxldouble *my_roots = *roots + (i * poly_degree);

      size_t found = process_single_combination(
        base_coeffs, num_base_coeffs, poly_degree, i,
        my_roots, poly_degree, use_lapack
      );

      (*num_distinct)[i] = found;
      total_roots += found;
    }
  }
#else
  for (size_t i = 0; i < num_combinations; i++) {
    // skip combinations if not divisible by skip value
    if (i % skip != 0) {
      // set num_distinct to 0 to maintain hue index
      (*num_distinct)[i] = 0;
      continue;
    }

    cxldouble *my_roots = *roots + (i * poly_degree);

    size_t found = process_single_combination(
      base_coeffs, num_base_coeffs, poly_degree, i,
      my_roots, poly_degree, use_lapack
    );

    (*num_distinct)[i] = found;
    total_roots += found;
  }
#endif

  return total_roots;
}

size_t polynomial_find_root_combinations(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, false, 1);
}

size_t polynomial_find_root_combinations_companion(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, true, 1);
}

size_t polynomial_find_root_combinations_skip(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, false, skip);
}

size_t polynomial_find_root_combinations_companion_skip(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, true, skip);
}