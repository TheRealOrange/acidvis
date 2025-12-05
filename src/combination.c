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

#include "polysolve.h"

#define UPDATE_INTERVAL      16
#define MAX_JT_ITERS_UPDATE  128

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

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

static int process_incremental_single_combination(
    const cxldouble *curr_coeffs,
    const cxldouble *prev_coeffs,
    cxldouble *work_deriv_coeffs,
    cxldouble *work_b,
    cxldouble *work_p,
    cxldouble *work_h,
    cxldouble *prev_roots,
    cxldouble *roots_out,
    size_t poly_degree) {
  // iterate N = degree times to take derivative
  // simultaneously use horner's method to evaluate
  // the polynomial derivative at the root for each root

  // setup horner's method intial vals
  // b_N = a_N
  // coeff N=deg+1
  work_deriv_coeffs[poly_degree - 1] = cxscalel(prev_coeffs[poly_degree], poly_degree);
  for (size_t i = 0; i < poly_degree; i++) {
    work_b[i] = work_deriv_coeffs[poly_degree-1];
  }

  for (size_t i = poly_degree-1; i > 0; i--) {
    // H(N-1) = (k(N) * N) * x ^ (N-1)
    work_deriv_coeffs[i - 1] = cxscalel(prev_coeffs[i], i);

    // horner's method evaluation for each root
    for (size_t j = 0; j < poly_degree; j++) {
      // b_N-1 = a_N-1 + b_N * z
      work_b[j] = cxaddl(work_deriv_coeffs[i-1], cxmull(work_b[j], prev_roots[j]));
    }
  }

  // work_b now contains the derivative evaluated at each root
  // calculate the sensitivity per coefficient
  for (size_t i = 0; i < poly_degree; i++) {
    cxldouble r = CX(1.0, 0.0);
    roots_out[i] = prev_roots[i];
    for (size_t k = 0; k < poly_degree+1; k++) {
      cxldouble delta_coeff = cxsubl(curr_coeffs[k],prev_coeffs[k]);
      // accumulate all the small coefficient changes
      roots_out[i] = cxadd(roots_out[i], cxdivl(cxmull(cxnegl(delta_coeff), r), work_b[i]));
      r = cxmull(r, r);
    }
  }

  int found = 0;
  // now roots_out contains the candidate roots for the new polynomial
  // for each root, try to find using just the 3rd stage of jenkins-traub
  polynomial_t *P = polynomial_from_coeffs(curr_coeffs, poly_degree + 1);

  // scale the polynomial (store the scale factor)
  long double sigma = polynomial_scale_for_roots(P);

  // make monic
  cxldouble lead = P->coeffs[P->degree];
  for (size_t i = 0; i <= P->degree; i++) {
    P->coeffs[i] = cxdivl(P->coeffs[i], lead);
  }

  // initialize H_0 = P'
  polynomial_t *H = H_0(P);
  polynomial_t *H_work;
  for (size_t i = 0; i < poly_degree; i++) {
    H_work = polynomial_copy(H);
    if (!H_work) {
      polynomial_free(P);
      polynomial_free(H);
      fprintf(stderr, "failed to allocate H_work\n");
      return 0;
    }

    // stage 3 search
    cxdouble scaled_root;
    cxldouble scaled_search = cxscaledivl(roots_out[i], sigma);
    jt_status res = iterate_find(H_work, P, work_p, work_h, scaled_search, &scaled_root, MAX_JT_ITERS_UPDATE);
    if (res == JT_CONVERGED) {
      found++;
      roots_out[i] = unscale_root(scaled_root, sigma);
    } else {
      // if there is any error or the root finding fails,
      // return 0 to indicate we want to fallback to full solve or
      // eigenvalue solve

      polynomial_free(H);
      polynomial_free(P);
      return 0;
    }
  }

  polynomial_free(H);
  polynomial_free(P);

  return found;
}

#ifdef HAVE_LAPACK
// batched LAPACK eigenvalue computation
// processes polynomials in chunks
static size_t process_batch_lapack(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    size_t *batch_idxes,
    size_t batch_size,
    cxldouble *roots_out,
    bool *combination_valid) {

  size_t n = poly_degree;
  size_t total_roots = 0;

  // allocate only for valid matrices
  cxdouble *matrices = malloc(batch_size * n * n * sizeof(cxdouble));
  cxdouble *eigenvalues = malloc(batch_size * n * sizeof(cxdouble));

  if (!matrices || !eigenvalues) {
    free(matrices);
    free(eigenvalues);
    return 0;
  }

  // build companion matrices in parallel
  int batch_i;  // msvc openmp requires signed int declared outside
#pragma omp parallel
  {
    // each thread allocates once
    cxldouble *coeffs = malloc((n + 1) * sizeof(cxldouble));

#pragma omp for schedule(static)
    for (batch_i = 0; batch_i < (int)batch_size; batch_i++) {
      if (!coeffs) continue;

      size_t global_i = batch_idxes[batch_i];

      // build polynomial coefficients
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
    }

    free(coeffs);
  }

  // compute eigenvalues only for matrices not skipped
  size_t successful = 0;
  if (batch_size > 0) {
    successful = compute_eigenvalues_batch(matrices, n, batch_size, eigenvalues);
    if (!successful) {
      fprintf(stderr, "failed to solve eigenvalue batch with degree=%zu, batch_size=%zu\n", n, batch_size);
    }
  }

  free(matrices);

  // convert results back to long double and store at correct global indices
  for (size_t it = 0; it < batch_size; it++) {
    size_t global_i = batch_idxes[it];

    // convert eigenvalues to long double
    cxldouble *roots = roots_out + global_i * n;
    cxdouble *evals = eigenvalues + (it * n);

    for (size_t j = 0; j < n; j++) {
      roots[j] = cx_to_cxl(evals[j]);
    }

    combination_valid[global_i] = true;
    total_roots += n;
  }

  free(eigenvalues);

  return total_roots;
}
#endif // HAVE_LAPACK

// process single combination and find its roots
static size_t process_single_combination(
    const cxldouble *coeffs,
    size_t poly_degree,
    cxldouble *roots_out,
    size_t max_roots,
    bool use_lapack) {

  size_t num_positions = poly_degree + 1;
  polynomial_t *work = polynomial_from_coeffs(coeffs, num_positions);

  if (!work) return 0;

#ifdef HAVE_LAPACK
  if (use_lapack) {
    polynomial_find_roots_companion(work, false);
  } else {
    polynomial_find_roots(work, false);
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

// simple non-cached version for full solves
// skip parameter: if > 1, only compute every skip-th combination
// skipped combinations have combination_valid[i] = false
size_t polynomial_find_root_combinations(
    const cxldouble *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble *roots,
    bool *combination_valid,
    int *since_last_update,
    size_t num_combinations,
    size_t skip,
    bool use_lapack) {

  for (size_t i = 0; i < num_combinations; i++) {
    since_last_update[i] = -1;  // force full solve
  }

  size_t result = polynomial_find_root_combinations_cached(
    base_coeffs, NULL,
    num_base_coeffs, poly_degree,
    roots, combination_valid, since_last_update,
    num_combinations, skip, use_lapack
  );
  return result;
}

// find roots for all N^(M+1) coefficient combinations
// skip parameter: if > 1, only compute every skip-th combination
// skipped combinations will have combination_valid[i] = false to maintain hue index
size_t polynomial_find_root_combinations_cached(
    const cxldouble *base_coeffs,
    const cxldouble *prev_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    cxldouble *roots,
    bool *combination_valid,
    int *since_last_update,
    size_t num_combinations,
    size_t skip, bool use_lapack) {

  bool incremental = true;
  if (!base_coeffs || num_base_coeffs == 0 || poly_degree == 0 || !roots || !combination_valid) {
    printf("invalid args\n");
    return 0;
  }

  // can't do incremental without previous state
  if (!prev_coeffs || !since_last_update) {
    incremental = false;
  }

  if (skip == 0) skip = 1;

  memset(combination_valid, 0, num_combinations * sizeof(bool));

  size_t num_to_process = CEILDIV(num_combinations, skip);
  size_t *fullsolve_idxes = malloc(num_to_process * sizeof(size_t));
  size_t *incremental_idxes = malloc(num_to_process * sizeof(size_t));
  int *roots_found_incremental = malloc(num_to_process * sizeof(int));

  if (!fullsolve_idxes || !incremental_idxes || !roots_found_incremental) {
    if (fullsolve_idxes) free(fullsolve_idxes);
    if (incremental_idxes) free(incremental_idxes);
    if (roots_found_incremental) free(roots_found_incremental);
    fprintf(stderr, "failed to allocate id arrays for num_combinations=%zu\n", num_combinations);
    return 0;
  }

  size_t total_roots = 0;
  int num_fullsolve = 0;
  int num_incremental = 0;

  // find out how many we need to fullsolve vs incremental solve
  for (size_t i = 0; i < num_combinations; i++) {
    if (i % skip == 0) {
      if (true || since_last_update[i] > UPDATE_INTERVAL || !incremental) {
        // need to fullsolve either because
        // it is too long since last update or
        // we are not requesting an incremental solve
        if (incremental) {
          since_last_update[i] = 0;
        } else {
          // if not incremental, we want to stagger the future updates
          since_last_update[i] = num_fullsolve % UPDATE_INTERVAL;
        }
        fullsolve_idxes[num_fullsolve] = i;
        num_fullsolve++;
      } else {
        since_last_update[i]++;
        incremental_idxes[num_incremental] = i;
        roots_found_incremental[num_incremental] = 0;
        num_incremental++;
      }
    } else {
      combination_valid[i] = false;
      since_last_update[i] = -1;
    }
  }

  size_t num_positions = poly_degree + 1;

  // incremental solve
#ifdef _OPENMP
  int it;
  #pragma omp parallel
  {
    // each thread allocates once
    cxldouble *prev_coeff_combination = malloc(num_positions * sizeof(cxldouble));
    cxldouble *curr_coeff_combination = malloc(num_positions * sizeof(cxldouble));
    cxldouble *work_deriv = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_b = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_p = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_h = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *prev_roots = calloc(num_positions, sizeof(cxldouble));

    #pragma omp for schedule(dynamic)
    for (it = 0; it < num_incremental; it++) {
      if (!prev_coeff_combination || !curr_coeff_combination ||
          !work_deriv || !work_b || !work_p || !work_h || !prev_roots) continue;

      size_t global_id = incremental_idxes[it];
      cxldouble *single_poly_roots = roots + (global_id * poly_degree);

      index_to_combination(global_id, num_base_coeffs, num_positions, base_coeffs, curr_coeff_combination);
      index_to_combination(global_id, num_base_coeffs, num_positions, prev_coeffs, prev_coeff_combination);

      // store a copy of the old roots
      memcpy(prev_roots, single_poly_roots, poly_degree * sizeof(cxldouble));
      size_t found_roots = process_incremental_single_combination(
        curr_coeff_combination, prev_coeff_combination,
        work_deriv, work_b, work_p, work_h,
        prev_roots, single_poly_roots, poly_degree
      );

      roots_found_incremental[it] = found_roots;
    }

    free(prev_coeff_combination);
    free(curr_coeff_combination);
    free(work_deriv);
    free(work_b);
    free(work_p);
    free(work_h);
    free(prev_roots);
  }
#else
  {
    cxldouble *prev_coeff_combination = malloc(num_positions * sizeof(cxldouble));
    cxldouble *curr_coeff_combination = malloc(num_positions * sizeof(cxldouble));
    cxldouble *work_deriv = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_b = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_p = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *work_h = calloc(poly_degree, sizeof(cxldouble));
    cxldouble *prev_roots = calloc(num_positions, sizeof(cxldouble));

    if (prev_coeff_combination && curr_coeff_combination &&
        work_deriv && work_b && work_p && work_h && prev_roots) {
      for (size_t it = 0; it < num_incremental; it++) {
        size_t global_id = incremental_idxes[it];
        cxldouble *single_poly_roots = roots + (global_id * poly_degree);

        index_to_combination(global_id, num_base_coeffs, num_positions, base_coeffs, curr_coeff_combination);
        index_to_combination(global_id, num_base_coeffs, num_positions, prev_coeffs, prev_coeff_combination);

        // store a copy of the old roots
        memcpy(prev_roots, single_poly_roots, poly_degree * sizeof(cxldouble));
        size_t found_roots = process_incremental_single_combination(
          curr_coeff_combination, prev_coeff_combination,
          work_deriv, work_b, work_p, work_h,
          prev_roots, single_poly_roots, poly_degree
        );

        roots_found_incremental[it] = found_roots;
      }
    }

    free(prev_coeff_combination);
    free(curr_coeff_combination);
    free(work_deriv);
    free(work_b);
    free(work_p);
    free(work_h);
    free(prev_roots);
  }
#endif

  // assign those with incomplete/no roots found back to fullsolve
  for (size_t i = 0; i < num_incremental; i++) {
    if (roots_found_incremental[i] == poly_degree) {
      combination_valid[incremental_idxes[i]] = true;
      total_roots += poly_degree;
    } else {
      // do not update since_last_solve here because we
      // want to maintain the staggered order
      fullsolve_idxes[num_fullsolve] = incremental_idxes[i];
      combination_valid[incremental_idxes[i]] = false;
      num_fullsolve++;
    }
  }

  free(incremental_idxes);
  free(roots_found_incremental);

  // use batched processing for LAPACK for the fullsolve indexes
#ifdef HAVE_LAPACK
  if (use_lapack) {
    size_t batch_size = MIN(MAX_BATCH_SIZE, num_fullsolve);

    // process in chunks
    for (size_t start = 0; start < num_fullsolve; start += batch_size) {
      size_t current_batch_size = batch_size;
      if (start + batch_size > num_fullsolve) {
        current_batch_size = num_fullsolve - start;
      }

      // get pointer to this chunk's indexes
      size_t *batch_idxes = fullsolve_idxes + start;

      size_t batch_total = process_batch_lapack(
        base_coeffs, num_base_coeffs, poly_degree,
        batch_idxes, current_batch_size,
        roots, combination_valid
      );

      total_roots += batch_total;
    }
  } else
#endif
  {
    // fallback: process fullsolve_idxes one-by-one
#ifdef _OPENMP
    int i;
    #pragma omp parallel
    {
      cxldouble *coeffs = malloc(num_positions * sizeof(cxldouble));

      #pragma omp for schedule(dynamic) reduction(+:total_roots)
      for (i = 0; i < num_fullsolve; i++) {
        size_t global_id = fullsolve_idxes[i];
        cxldouble *my_roots = roots + (global_id * poly_degree);

        index_to_combination(global_id, num_base_coeffs, num_positions, base_coeffs, coeffs);

        size_t found = process_single_combination(
          coeffs, poly_degree,
          my_roots, poly_degree, use_lapack
        );

        combination_valid[global_id] = found == poly_degree;
        total_roots += found;
      }

      free(coeffs);
    }
#else
    cxldouble *coeffs = malloc(num_positions * sizeof(cxldouble));
    if (coeffs) {
      for (size_t i = 0; i < num_fullsolve; i++) {
        size_t global_id = fullsolve_idxes[i];
        cxldouble *my_roots = roots + (global_id * poly_degree);

        index_to_combination(global_id, num_base_coeffs, num_positions, base_coeffs, coeffs);

        size_t found = process_single_combination(
          coeffs, poly_degree,
          my_roots, poly_degree, use_lapack
        );

        combination_valid[global_id] = found == poly_degree;
        total_roots += found;
      }
    }
    free(coeffs);
#endif
  }

  free(fullsolve_idxes);

  return total_roots;
}