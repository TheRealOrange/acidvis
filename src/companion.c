#include "companion.h"

#ifdef HAVE_LAPACK

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// zgeev: Computes eigenvalues (and optionally eigenvectors) of a complex double matrix
//
// JOBVL: 'N' = don't compute left eigenvectors
// JOBVR: 'N' = don't compute right eigenvectors
// N:     Order of the matrix
// A:     Complex matrix (destroyed on output)
// LDA:   Leading dimension of A
// W:     Output eigenvalues
// VL:    Left eigenvectors (not referenced if JOBVL='N')
// LDVL:  Leading dimension of VL
// VR:    Right eigenvectors (not referenced if JOBVR='N')
// LDVR:  Leading dimension of VR
// WORK:  Workspace array
// LWORK: Size of WORK (-1 for query)
// RWORK: Real workspace array
// INFO:  0 = success, <0 = illegal argument, >0 = failed to converge

#ifdef __APPLE__
// macOS Accelerate framework uses different naming
#include <Accelerate/Accelerate.h>
#define LAPACK_zgeev zgeev_
#else
// fortran interface
extern void zgeev_(const char *jobvl, const char *jobvr, const int *n,
                   double _Complex *a, const int *lda,
                   double _Complex *w,
                   double _Complex *vl, const int *ldvl,
                   double _Complex *vr, const int *ldvr,
                   double _Complex *work, const int *lwork,
                   double *rwork, int *info);
#define LAPACK_zgeev zgeev_
#endif

#define COMPANION_EPSILON 1e-12
#define REL_EPSILON 1e-15

static bool complex_approx_equal_d(complex double a, complex double b) {
  double diff = cabs(a - b);
  double mag = fmax(cabs(a), cabs(b));
  return diff < (DBL_EPSILON + REL_EPSILON * mag);
}

// Build the companion matrix for a monic polynomial
// For p(x) = x^n + c_{n-1}*x^{n-1} + ... + c_1*x + c_0
// The companion matrix (Frobenius form) is:
//
// [ 0  0  0  ...  0  -c_0   ]
// [ 1  0  0  ...  0  -c_1   ]
// [ 0  1  0  ...  0  -c_2   ]
// [ ...                      ]
// [ 0  0  0  ...  1  -c_{n-1}]
//
// The eigenvalues of this matrix are exactly the roots of p(x)
//
// Matrix is stored in column-major order for LAPACK
static complex double *build_companion_matrix(polynomial_t *poly, size_t n) {
  // n x n matrix stored in column-major order
  complex double *matrix = calloc(n * n, sizeof(complex double));
  if (!matrix) return NULL;

  // Get the leading coefficient for normalization (make monic)
  complex long double lead = poly->coeffs[n];

  // Fill the subdiagonal with 1s
  // In column-major: element (i,j) is at index i + j*n
  for (size_t i = 1; i < n; i++) {
    matrix[i + (i - 1) * n] = 1.0;
  }

  // Fill the last column with -c_i / c_n (negative normalized coefficients)
  for (size_t i = 0; i < n; i++) {
    complex long double coeff = poly->coeffs[i] / lead;
    matrix[i + (n - 1) * n] = (complex double) (-coeff);
  }

  return matrix;
}

// Compute eigenvalues using LAPACK's zgeev
static bool compute_eigenvalues(complex double *matrix, size_t n, complex double *eigenvalues) {
  int info;
  int N = (int) n;

  // fortran interface
  char jobvl = 'N';
  char jobvr = 'N';
  int ldvl = 1;
  int ldvr = 1;

  // workspace query
  complex double work_query;
  int lwork = -1;
  double *rwork = malloc(2 * n * sizeof(double));
  if (!rwork) return false;

  LAPACK_zgeev(&jobvl, &jobvr, &N, matrix, &N, eigenvalues,
               nullptr, &ldvl, nullptr, &ldvr,
               &work_query, &lwork, rwork, &info);

  if (info != 0) {
    free(rwork);
    return false;
  }

  lwork = (int) creal(work_query);
  complex double *work = malloc(lwork * sizeof(complex double));
  if (!work) {
    free(rwork);
    return false;
  }

  // computation
  LAPACK_zgeev(&jobvl, &jobvr, &N, matrix, &N, eigenvalues,
               nullptr, &ldvl, nullptr, &ldvr,
               work, &lwork, rwork, &info);

  free(work);
  free(rwork);

  return (info == 0);
}

// batched eigenvalue computation with memory awareness
// processes multiple companion matrices in parallel
size_t compute_eigenvalues_batch(
  complex double *matrices, // input: batch_size Ã— n Ã— n matrices (contiguous)
  size_t n, // matrix dimension
  size_t batch_size, // number of matrices
  complex double *eigenvalues) {
  // output: batch_size Ã— n eigenvalues (contiguous)

  if (!matrices || !eigenvalues || n == 0 || batch_size == 0) {
    return 0;
  }

  int N = (int) n;
  char jobvl = 'N';
  char jobvr = 'N';
  int ldvl = 1;
  int ldvr = 1;

  // do workspace query using the first matrix
  complex double work_query;
  int lwork = -1;
  double *rwork_query = malloc(2 * n * sizeof(double));
  if (!rwork_query) return 0;

  int info;
  LAPACK_zgeev(&jobvl, &jobvr, &N, matrices, &N, eigenvalues,
               nullptr, &ldvl, nullptr, &ldvr,
               &work_query, &lwork, rwork_query, &info);

  free(rwork_query);

  if (info != 0) {
    return 0;
  }

  lwork = (int) creal(work_query);

  // now process all matrices in parallel with thread-local workspaces
  size_t successful = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:successful)
  {
    // each thread gets its own workspace
    double *rwork = malloc(2 * n * sizeof(double));
    complex double *work = malloc(lwork * sizeof(complex double));

    if (rwork && work) {
#pragma omp for schedule(dynamic, 4)
      for (size_t i = 0; i < batch_size; i++) {
        complex double *matrix = matrices + (i * n * n);
        complex double *evals = eigenvalues + (i * n);

        // copy matrix since LAPACK destroys it
        complex double *matrix_copy = malloc(n * n * sizeof(complex double));
        if (!matrix_copy) continue;

        memcpy(matrix_copy, matrix, n * n * sizeof(complex double));

        // compute eigenvalues
        int thread_info;
        LAPACK_zgeev(&jobvl, &jobvr, &N, matrix_copy, &N, evals,
                     nullptr, &ldvl, nullptr, &ldvr,
                     work, &lwork, rwork, &thread_info);

        free(matrix_copy);

        if (thread_info == 0) {
          successful++;
        }
      }
    }

    free(work);
    free(rwork);
  }
#else
  // sequential fallback
  double *rwork = malloc(2 * n * sizeof(double));
  complex double *work = malloc(lwork * sizeof(complex double));

  if (rwork && work) {
    for (size_t i = 0; i < batch_size; i++) {
      complex
      double *matrix = matrices + (i * n * n);
      complex
      double *evals = eigenvalues + (i * n);

      complex
      double *matrix_copy = malloc(n * n * sizeof(complex double)
      )
      ;
      if (!matrix_copy) continue;

      memcpy(matrix_copy, matrix, n * n * sizeof(complex double)
      )
      ;

      int thread_info;
      LAPACK_zgeev(&jobvl, &jobvr, &N, matrix_copy, &N, evals,
                   nullptr, &ldvl, nullptr, &ldvr,
                   work, &lwork, rwork, &thread_info);

      free(matrix_copy);

      if (thread_info == 0) {
        successful++;
      }
    }
  }

  free(work);
  free(rwork);
#endif

  return successful;
}

// Polish a root using Newton's method
static complex long double polish_root_companion(polynomial_t *poly, complex long double z) {
  const int MAX_ITERS = 30;

  for (int iter = 0; iter < MAX_ITERS; iter++) {
    // Evaluate P(z) and P'(z) using Horner's method
    complex long double p_val = poly->coeffs[poly->degree];
    complex long double dp_val = 0;

    for (size_t i = poly->degree; i > 0; i--) {
      dp_val = dp_val * z + p_val;
      p_val = poly->coeffs[i - 1] + p_val * z;
    }

    if (cabsl(dp_val) < LDBL_EPSILON) break;

    complex long double delta = p_val / dp_val;
    z = z - delta;

    if (cabsl(delta) < REL_EPSILON * fmaxl(1.0L, cabsl(z))) break;
  }

  return z;
}

bool _polynomial_find_roots_companion(polynomial_t *poly,
                                      complex long double *roots,
                                      size_t *num_roots) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0 || !roots || !num_roots) {
    return false;
  }

  size_t n = poly->degree;
  *num_roots = 0;

  // Handle degree 1 directly: ax + b = 0 => x = -b/a
  if (n == 1) {
    roots[0] = -poly->coeffs[0] / poly->coeffs[1];
    *num_roots = 1;
    return true;
  }

  // Handle degree 2 with quadratic formula for better precision
  if (n == 2) {
    complex long double a = poly->coeffs[2];
    complex long double b = poly->coeffs[1];
    complex long double c = poly->coeffs[0];
    complex long double disc = csqrtl(b * b - 4.0L * a * c);
    roots[0] = (-b + disc) / (2.0L * a);
    roots[1] = (-b - disc) / (2.0L * a);
    *num_roots = 2;
    return true;
  }

  // Build companion matrix
  complex double *matrix = build_companion_matrix(poly, n);
  if (!matrix) return false;

  // Allocate eigenvalue array
  complex double *eigenvalues = malloc(n * sizeof(complex double));
  if (!eigenvalues) {
    free(matrix);
    return false;
  }

  // Compute eigenvalues
  bool success = compute_eigenvalues(matrix, n, eigenvalues);
  free(matrix);

  if (!success) {
    free(eigenvalues);
    return false;
  }

  // Convert eigenvalues to long double and polish them
  for (size_t i = 0; i < n; i++) {
    complex long double raw_root = (complex long double) eigenvalues[i];
    roots[i] = polish_root_companion(poly, raw_root);
  }

  free(eigenvalues);
  *num_roots = n;
  return true;
}

// Compare function for sorting complex numbers
static int complex_compare_d(const void *a, const void *b) {
  complex long double ca = *(const complex long double *) a;
  complex long double cb = *(const complex long double *) b;

  long double real_diff = creall(ca) - creall(cb);
  if (fabsl(real_diff) > LDBL_EPSILON) {
    return (real_diff > 0) - (real_diff < 0);
  }

  long double imag_diff = cimagl(ca) - cimagl(cb);
  return (imag_diff > 0) - (imag_diff < 0);
}

bool polynomial_find_roots_companion(polynomial_t *poly) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0) {
    return false;
  }

  complex long double *raw_roots = malloc(poly->degree * sizeof(complex long double));
  if (!raw_roots) return false;

  size_t num_roots;
  bool success = _polynomial_find_roots_companion(poly, raw_roots, &num_roots);

  if (!success || num_roots == 0) {
    free(raw_roots);
    return false;
  }

  // Sort roots for deduplication
  qsort(raw_roots, num_roots, sizeof(complex long double), complex_compare_d);

  // Deduplicate and compute multiplicities
  size_t distinct_count = 0;
  complex long double prev_root = 0;

  for (size_t i = 0; i < num_roots; i++) {
    if (distinct_count == 0 ||
        !complex_approx_equal_d((complex double) raw_roots[i], (complex double) prev_root)) {
      // New distinct root
      poly->roots[distinct_count] = raw_roots[i];
      poly->multiplicity[distinct_count] = 1;
      prev_root = raw_roots[i];
      distinct_count++;
    } else {
      // Repeated root - increment multiplicity
      poly->multiplicity[distinct_count - 1]++;
    }
  }

  poly->num_distinct_roots = distinct_count;
  poly->roots_valid = true;

  free(raw_roots);
  return true;
}

#endif // HAVE_LAPACK