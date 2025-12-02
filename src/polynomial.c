//
// Created by Lin Yicheng on 30/11/25.
//

#include "polynomial.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "companion.h"
#include "polysolve.h"
#include "util.h"

#define REL_EPSILON 1e-20

static bool complex_approx_equal(complex long double a, complex long double b) {
  long double diff = cabsl(a - b);
  long double mag = fmaxl(cabsl(a), cabsl(b));
  return diff < (LDBL_EPSILON + REL_EPSILON * mag);
}

// need consistent ordering, so compare by real part first
// and then compare by imaginary part
static int complex_compare(const void *a, const void *b) {
  complex long double ca = *(const complex long double *)a;
  complex long double cb = *(const complex long double *)b;

  long double real_diff = creall(ca) - creall(cb);
  if (fabsl(real_diff) > 0)
    return (real_diff > 0) - (real_diff < 0);

  long double imag_diff = cimagl(ca) - cimagl(cb);
  if (fabsl(imag_diff) > 0)
    return (imag_diff > 0) - (imag_diff < 0);

  return 0;
}

static void expand_from_roots(complex long double *coeffs,
                              const complex long double *roots,
                              const size_t *mults,
                              size_t distinct_roots,
                              size_t coeffs_size) {
  size_t curr_deg = 0;
  for (size_t i = 0; i < coeffs_size; i++) coeffs[i] = 0;
  coeffs[0] = 1.0;
  for (size_t i = 0; i < distinct_roots; i++) {
    // multipy the current expanded form by
    // (x - c) for each root
    for (size_t m = 0; m < mults[i]; m++) {
      // for each root, account for multiplicity
      // by just multiplying repeatedly

      // degree increases by one
      curr_deg++;
      // first coefficient takse from previous degree implicitly
      // since coeffs is all 0 initially
      for (size_t j = curr_deg; j > 0; j--) {
        // for each coefficient, subtract the current one
        // multiplied by the root (x - c) -> k(N) * (-c) + k(N-1)
        coeffs[j] = -roots[i] * coeffs[j] + coeffs[j-1];
      }

      // last coefficient is just multiplied by neg of the root
      // k(0) * (-c)
      coeffs[0] *= -roots[i];
    }
  }
}

polynomial_t *polynomial_new(size_t degree) {
  polynomial_t *poly = malloc(sizeof(polynomial_t));
  if (!poly) return nullptr;

  poly->degree = degree;
  poly->coeffs = calloc(degree+1, sizeof(complex long double));
  poly->roots = calloc(degree, sizeof(complex long double));
  poly->multiplicity = calloc(degree, sizeof(size_t));

  if (!poly->coeffs || !poly->roots || !poly->multiplicity) {
    // falied to allocate somehow
    polynomial_free(poly);
    return nullptr;
  }

  // coefficients not set yet
  poly->coeffs_valid = false;

  // roots not found yet
  poly->num_distinct_roots = 0;
  poly->roots_valid = false;

  return poly;
}

polynomial_t *polynomial_from_roots(complex long double *roots, size_t num_roots, bool dedup) {
  if (!roots || !num_roots) return nullptr;
  size_t num_dist_roots = 0;
  // find the number of distinct roots
  complex long double *distinct_roots;
  size_t *multiplicity;
  distinct_roots = calloc(num_roots, sizeof(complex long double));
  multiplicity = calloc(num_roots, sizeof(size_t));
  if (!distinct_roots || !multiplicity) {
    if (distinct_roots) free(distinct_roots);
    if (multiplicity) free (multiplicity);
    return nullptr;
  }

  if (dedup) {
    // assuming that repeated roots show up in the roots more than once
    // so we copy the roots and sort them
    complex long double *sorted_roots;
    sorted_roots = calloc(num_roots, sizeof(complex long double));
    if (!sorted_roots) return nullptr;

    memcpy(sorted_roots, roots, num_roots * sizeof(complex long double));
    qsort(sorted_roots, num_roots, sizeof(complex long double), complex_compare);

    complex long double curr_root = 0;
    for (size_t i = 0; i < num_roots; i++) {
      // find repeated roots
      if (num_dist_roots == 0 || !complex_approx_equal(sorted_roots[i], curr_root)) {
        // first root or different from current root
        curr_root = sorted_roots[i];
        num_dist_roots++;

        distinct_roots[num_dist_roots-1] = sorted_roots[i];
        multiplicity[num_dist_roots-1] = 1;
      } else {
        // same as current root
        // increase multiplicity
        multiplicity[num_dist_roots-1]++;
      }
    }

    free(sorted_roots);
  } else {
    num_dist_roots = num_roots;
    for (size_t i = 0; i < num_roots; i++) {
      distinct_roots[i] = roots[i];
      multiplicity[i] = 1;
    }
  }

  polynomial_t *poly = polynomial_from_dis_roots(distinct_roots, multiplicity, num_dist_roots);
  free(multiplicity);
  free(distinct_roots);

  return poly;
}

polynomial_t *polynomial_from_dis_roots(complex long double *distinct_roots, size_t *mult, size_t num_dist_roots) {
  if (!distinct_roots || num_dist_roots < 1) return nullptr;

  // calculate degree from the number of roots and multiplicity
  size_t degree = 0;
  for (size_t i = 0; i < num_dist_roots; i++) degree += mult[i];
  polynomial_t *poly = polynomial_new(degree);
  if (!poly) return nullptr;

  memcpy(poly->roots, distinct_roots, num_dist_roots * sizeof(complex long double));
  memcpy(poly->multiplicity, mult, num_dist_roots * sizeof(size_t));
  poly->num_distinct_roots = num_dist_roots;
  poly->roots_valid = true;

  // calculate coefficients from roots
  expand_from_roots(poly->coeffs, poly->roots, poly->multiplicity,
    poly->num_distinct_roots, poly->degree+1);
  poly->coeffs_valid = true;

  return poly;
}

polynomial_t *polynomial_from_coeffs(complex long double *coeffs, size_t num_coeffs) {
  if (!coeffs || num_coeffs < 1) return nullptr;

  // assuming number of coefficients-1 is the degree
  polynomial_t *poly = polynomial_new(num_coeffs-1);
  if (!poly) return nullptr;

  memcpy(poly->coeffs, coeffs, num_coeffs * sizeof(complex long double));
  poly->coeffs_valid = true;

  return poly;
}

void polynomial_free(polynomial_t *poly) {
  if (!poly) return;
  free(poly->coeffs);
  free(poly->roots);
  free(poly->multiplicity);

  free(poly);
}

polynomial_t *polynomial_copy(polynomial_t *poly) {
  if (!poly) return nullptr;

  polynomial_t *copy = polynomial_new(poly->degree);
  if (!copy) return nullptr;

  memcpy(copy->coeffs, poly->coeffs, (poly->degree+1) * sizeof(complex long double));
  memcpy(copy->roots, poly->roots, poly->num_distinct_roots * sizeof(complex long double));
  memcpy(copy->multiplicity, poly->multiplicity, poly->num_distinct_roots * sizeof(size_t));

  copy->num_distinct_roots = poly->num_distinct_roots;
  copy->roots_valid = poly->roots_valid;
  copy->coeffs_valid = poly->coeffs_valid;
  return copy;
}

// horner's method for evaluating the polynomial
complex long double polynomial_eval(polynomial_t *poly, complex long double z) {
  // b_N = a_N
  // coeff N=deg+1
  complex long double b = poly->coeffs[poly->degree];
  for (size_t i = poly->degree; i > 0; i--) {
    // b_N-1 = a_N-1 + b_N * z
    b = poly->coeffs[i-1] + b * z;
  }
  return b;
}

// find roots using the scaled Jenkins-Traub algorithm
bool polynomial_find_roots(polynomial_t *poly) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0) return false;

  complex long double *roots = malloc(poly->degree * sizeof(complex long double));
  if (!roots) return false;

  size_t num_roots;
  bool success = polynomial_find_roots_scaled(poly, roots, &num_roots);

  if (success && num_roots > 0) {
    // use constructor to deduplicate roots
    polynomial_t *temp = polynomial_from_roots(roots, num_roots, true);
    if (temp) {
      // copy deduplicated roots and multiplicities
      memcpy(poly->roots, temp->roots, temp->num_distinct_roots * sizeof(complex long double));
      memcpy(poly->multiplicity, temp->multiplicity, temp->num_distinct_roots * sizeof(size_t));
      poly->num_distinct_roots = temp->num_distinct_roots;
      poly->roots_valid = true;
      polynomial_free(temp);
    } else {
      // deduplication failed, copy raw roots with multiplicity 1
      memcpy(poly->roots, roots, num_roots * sizeof(complex long double));
      for (size_t i = 0; i < num_roots; i++) {
        poly->multiplicity[i] = 1;
      }
      poly->num_distinct_roots = num_roots;
      poly->roots_valid = true;
    }
  }

  free(roots);
  return success;
}


// convert combination index to coefficient assignment
// treats index as base-N number where each digit picks which base coeff to use
static void index_to_combination(size_t index, size_t num_base_coeffs, size_t num_positions,
                                  const complex long double *base_coeffs,
                                  complex long double *out_coeffs) {
  for (size_t pos = 0; pos < num_positions; pos++) {
    size_t which_base = index % num_base_coeffs;
    out_coeffs[pos] = base_coeffs[which_base];
    index /= num_base_coeffs;
  }
}

#ifdef HAVE_LAPACK
// memory-aware batched LAPACK eigenvalue computation
// processes polynomials in chunks to avoid OOM
// delegates actual LAPACK calls to companion.c
static size_t process_batch_lapack(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    size_t start_idx,
    size_t batch_size,
    complex long double *roots_out,
    size_t *num_distinct_out,
    size_t skip) {

  size_t n = poly_degree;
  size_t total_roots = 0;

  // allocate batch arrays for this chunk
  // first pass: build index mapping (must be serial to avoid race conditions)
  size_t *batch_to_valid = malloc(batch_size * sizeof(size_t));
  if (!batch_to_valid) {
    return 0;
  }

  size_t num_valid = 0;
  for (size_t batch_i = 0; batch_i < batch_size; batch_i++) {
    size_t global_i = start_idx + batch_i;
    if (global_i % skip == 0) {
      batch_to_valid[batch_i] = num_valid;
      num_valid++;
    } else {
      batch_to_valid[batch_i] = (size_t)-1;  // mark as invalid
      num_distinct_out[batch_i] = 0;
    }
  }

  // allocate only for valid matrices
  complex double *matrices = malloc(num_valid * n * n * sizeof(complex double));
  complex double *eigenvalues = malloc(num_valid * n * sizeof(complex double));

  if (!matrices || !eigenvalues) {
    free(matrices);
    free(eigenvalues);
    free(batch_to_valid);
    return 0;
  }

  // build companion matrices in parallel (now safe - each thread has unique valid_i)
  #pragma omp parallel for schedule(static)
  for (size_t batch_i = 0; batch_i < batch_size; batch_i++) {
    size_t global_i = start_idx + batch_i;

    // skip if not divisible by skip value
    if (global_i % skip != 0) {
      continue;
    }

    size_t valid_i = batch_to_valid[batch_i];

    // build polynomial coefficients
    complex long double coeffs[n + 1];
    index_to_combination(global_i, num_base_coeffs, n + 1, base_coeffs, coeffs);

    // normalize to monic
    complex long double lead = coeffs[n];
    for (size_t i = 0; i <= n; i++) {
      coeffs[i] /= lead;
    }

    // build companion matrix in column-major order at compacted position
    complex double *matrix = matrices + (valid_i * n * n);
    memset(matrix, 0, n * n * sizeof(complex double));

    // subdiagonal: 1s
    for (size_t i = 1; i < n; i++) {
      matrix[i + (i-1) * n] = 1.0;
    }

    // last column: -c_i (negated normalized coefficients)
    for (size_t i = 0; i < n; i++) {
      matrix[i + (n-1) * n] = (complex double)(-coeffs[i]);
    }
  }

  // compute eigenvalues only for valid matrices
  size_t successful = 0;
  if (num_valid > 0) {
    successful = compute_eigenvalues_batch(matrices, n, num_valid, eigenvalues);
  }

  // convert results back to long double and store at correct batch indices
  for (size_t batch_i = 0; batch_i < batch_size; batch_i++) {
    size_t global_i = start_idx + batch_i;

    if (global_i % skip != 0) {
      continue;
    }

    size_t valid_i = batch_to_valid[batch_i];

    // convert eigenvalues to long double
    complex long double *roots = roots_out + (batch_i * n);
    complex double *evals = eigenvalues + (valid_i * n);

    for (size_t j = 0; j < n; j++) {
      roots[j] = (complex long double)evals[j];
    }

    num_distinct_out[batch_i] = n;
    total_roots += n;
  }

  free(batch_to_valid);

  free(matrices);
  free(eigenvalues);

  return total_roots;
}
#endif // HAVE_LAPACK

// process single combination and find its roots
static size_t process_single_combination(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    size_t combination_index,
    complex long double *roots_out,
    size_t max_roots,
    bool use_lapack) {

  size_t num_positions = poly_degree + 1;

  complex long double *coeffs = malloc(num_positions * sizeof(complex long double));
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

  memcpy(roots_out, work->roots, num_distinct * sizeof(complex long double));

  polynomial_free(work);
  return num_distinct;
}

// find roots for all N^(M+1) coefficient combinations
// skip parameter: if > 1, only compute every skip-th combination
// skipped combinations will have num_distinct[i] = 0 to maintain hue index
static size_t polynomial_find_root_combinations_helper(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    complex long double **roots,
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

    // estimate memory per polynomial:
    // - companion matrix: n*n*sizeof(complex double) = n*n*16 bytes
    // - eigenvalues: n*sizeof(complex double) = n*16 bytes
    // - LAPACK workspace per thread: ~n*n*16 bytes
    size_t bytes_per_poly = n * n * 16 + n * 16;

    // limit to reasonable batch sizes
    size_t max_batch_size = MAX_BATCH_SIZE;
    size_t target_batch_bytes = MAX_BATCH_MEM_MB * 1024 * 1024;
    size_t batch_size = target_batch_bytes / bytes_per_poly;

    if (batch_size > max_batch_size) batch_size = max_batch_size;
    if (batch_size > num_combinations) batch_size = num_combinations;
    if (batch_size < 4) batch_size = 4;  // minimum batch size

    // process in chunks
    for (size_t start = 0; start < num_combinations; start += batch_size) {
      size_t current_batch_size = batch_size;
      if (start + batch_size > num_combinations) {
        current_batch_size = num_combinations - start;
      }

      // get pointers to this chunk's output
      complex long double *batch_roots = *roots + (start * poly_degree);
      size_t *batch_num_distinct = *num_distinct + start;

      size_t batch_total = process_batch_lapack(
        base_coeffs, num_base_coeffs, poly_degree,
        start, current_batch_size,
        batch_roots, batch_num_distinct, skip
      );

      total_roots += batch_total;
    }

    return total_roots;
  }
#endif

  // fallback, process one-by-one, for non-LAPACK or small batches
#ifdef _OPENMP
  #pragma omp parallel reduction(+:total_roots)
  {
    #pragma omp for schedule(dynamic, 64)
    for (size_t i = 0; i < num_combinations; i++) {
      // skip combinations if not divisible by skip value
      if (i % skip != 0) {
        // set num_distinct to 0 to maintain hue index
        (*num_distinct)[i] = 0;
        continue;
      }

      complex long double *my_roots = *roots + (i * poly_degree);

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

    complex long double *my_roots = *roots + (i * poly_degree);

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
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    complex long double **roots,
    size_t **num_distinct,
    size_t num_combinations) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, false, 1);
}

size_t polynomial_find_root_combinations_companion(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    complex long double **roots,
    size_t **num_distinct,
    size_t num_combinations) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, true, 1);
}

size_t polynomial_find_root_combinations_skip(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    complex long double **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, false, skip);
}

size_t polynomial_find_root_combinations_companion_skip(
    const complex long double *base_coeffs,
    size_t num_base_coeffs,
    size_t poly_degree,
    complex long double **roots,
    size_t **num_distinct,
    size_t num_combinations,
    size_t skip) {
  return polynomial_find_root_combinations_helper(
    base_coeffs, num_base_coeffs, poly_degree,
    roots, num_distinct, num_combinations, true, skip);
}