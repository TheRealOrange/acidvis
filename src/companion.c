#include "companion.h"

#ifdef HAVE_LAPACK

#include "compat_complex.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// zgeev: Computes eigenvalues (and optionally eigenvectors) of a cxdouble matrix
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
// _Dcomplex on msvc, double _Complex elsewhere
#if defined(_MSC_VER) && !defined(__clang__)
extern void zgeev_(const char *jobvl, const char *jobvr, const int *n,
                   _Dcomplex *a, const int *lda,
                   _Dcomplex *w,
                   _Dcomplex *vl, const int *ldvl,
                   _Dcomplex *vr, const int *ldvr,
                   _Dcomplex *work, const int *lwork,
                   double *rwork, int *info);
#else
extern void zgeev_(const char *jobvl, const char *jobvr, const int *n,
                   double _Complex *a, const int *lda,
                   double _Complex *w,
                   double _Complex *vl, const int *ldvl,
                   double _Complex *vr, const int *ldvr,
                   double _Complex *work, const int *lwork,
                   double *rwork, int *info);
#endif
#define LAPACK_zgeev zgeev_
#endif

#define COMPANION_EPSILON 1e-12
#define REL_EPSILON 1e-15

static bool complex_approx_equal_d(cxdouble a, cxdouble b) {
  double diff = cxabs(cxsub(a, b));
  double mag = fmax(cxabs(a), cxabs(b));
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
static cxdouble *build_companion_matrix(polynomial_t *poly, size_t n) {
  // n x n matrix stored in column-major order
  cxdouble *matrix = calloc(n * n, sizeof(cxdouble));
  if (!matrix) return NULL;

  // Get the leading coefficient for normalization (make monic)
  cxldouble lead = poly->coeffs[n];

  // Fill the subdiagonal with 1s
  // In column-major: element (i,j) is at index i + j*n
  for (size_t i = 1; i < n; i++) {
    matrix[i + (i - 1) * n] = CXL(1.0, 0.0);
  }

  // Fill the last column with -c_i / c_n (negative normalized coefficients)
  for (size_t i = 0; i < n; i++) {
    cxldouble coeff = cxdivl(poly->coeffs[i], lead);
    matrix[i + (n - 1) * n] = cxnegl(coeff);
  }

  return matrix;
}

// Compute eigenvalues using LAPACK's zgeev
static bool compute_eigenvalues(cxdouble *matrix, size_t n, cxdouble *eigenvalues) {
  int info;
  int N = (int) n;

  // fortran interface
  char jobvl = 'N';
  char jobvr = 'N';
  int ldvl = 1;
  int ldvr = 1;

  // workspace query
  cxdouble work_query;
  int lwork = -1;
  double *rwork = malloc(2 * n * sizeof(double));
  if (!rwork) return false;

  LAPACK_zgeev(&jobvl, &jobvr, &N, matrix, &N, eigenvalues,
               NULL, &ldvl, NULL, &ldvr,
               &work_query, &lwork, rwork, &info);

  if (info != 0) {
    free(rwork);
    return false;
  }

  lwork = (int) cxreal(work_query);
  cxdouble *work = malloc(lwork * sizeof(cxdouble));
  if (!work) {
    free(rwork);
    return false;
  }

  // computation
  LAPACK_zgeev(&jobvl, &jobvr, &N, matrix, &N, eigenvalues,
               NULL, &ldvl, NULL, &ldvr,
               work, &lwork, rwork, &info);

  free(work);
  free(rwork);

  return (info == 0);
}

// batched eigenvalue computation
// processes multiple companion matrices in parallel
size_t compute_eigenvalues_batch(
  cxdouble *matrices, // input: batch_size × n × n matrices (contiguous)
  size_t n, // matrix dimension
  size_t batch_size, // number of matrices
  cxdouble *eigenvalues) {
  // output: batch_size × n eigenvalues (contiguous)

  if (!matrices || !eigenvalues || n == 0 || batch_size == 0) {
    return 0;
  }

  int N = (int) n;
  char jobvl = 'N';
  char jobvr = 'N';
  int ldvl = 1;
  int ldvr = 1;

  // do workspace query using the first matrix
  cxdouble work_query;
  int lwork = -1;
  double *rwork_query = malloc(2 * n * sizeof(double));
  if (!rwork_query) return 0;

  int info;
  LAPACK_zgeev(&jobvl, &jobvr, &N, matrices, &N, eigenvalues,
               NULL, &ldvl, NULL, &ldvr,
               &work_query, &lwork, rwork_query, &info);

  free(rwork_query);

  if (info != 0) {
    return 0;
  }

  lwork = (int) cxreal(work_query);

  // now process all matrices in parallel with thread-local workspaces
  size_t successful = 0;

#ifdef _OPENMP
  int i;  // msvc openmp requires signed int declared outside
#pragma omp parallel reduction(+:successful)
  {
    // each thread gets its own workspace
    double *rwork = malloc(2 * n * sizeof(double));
    cxdouble *work = malloc(lwork * sizeof(cxdouble));

    if (rwork && work) {
#pragma omp for schedule(dynamic, 4)
      for (i = 0; i < (int)batch_size; i++) {
        cxdouble *matrix = matrices + (i * n * n);
        cxdouble *evals = eigenvalues + (i * n);

        // copy matrix since LAPACK destroys it
        cxdouble *matrix_copy = malloc(n * n * sizeof(cxdouble));
        if (!matrix_copy) continue;

        memcpy(matrix_copy, matrix, n * n * sizeof(cxdouble));

        // compute eigenvalues
        int thread_info;
        LAPACK_zgeev(&jobvl, &jobvr, &N, matrix_copy, &N, evals,
                     NULL, &ldvl, NULL, &ldvr,
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
  cxdouble *work = malloc(lwork * sizeof(cxdouble));

  if (rwork && work) {
    for (size_t i = 0; i < batch_size; i++) {
      cxdouble *matrix = matrices + (i * n * n);
      cxdouble *evals = eigenvalues + (i * n);

      cxdouble *matrix_copy = malloc(n * n * sizeof(cxdouble));
      if (!matrix_copy) continue;

      memcpy(matrix_copy, matrix, n * n * sizeof(cxdouble));

      int thread_info;
      LAPACK_zgeev(&jobvl, &jobvr, &N, matrix_copy, &N, evals,
                   NULL, &ldvl, NULL, &ldvr,
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
static cxldouble polish_root_companion(polynomial_t *poly, cxldouble z) {
  const int MAX_ITERS = 30;

  for (int iter = 0; iter < MAX_ITERS; iter++) {
    // Evaluate P(z) and P'(z) using Horner's method
    cxldouble p_val = poly->coeffs[poly->degree];
    cxldouble dp_val = CXL(0.0, 0.0);

    for (size_t i = poly->degree; i > 0; i--) {
      dp_val = cxaddl(cxmull(dp_val, z), p_val);
      p_val = cxaddl(poly->coeffs[i - 1], cxmull(p_val, z));
    }

    if (cxabsl(dp_val) < LDBL_EPSILON) break;

    cxldouble delta = cxdivl(p_val, dp_val);
    z = cxsubl(z, delta);

    if (cxabsl(delta) < REL_EPSILON * fmaxl(1.0L, cxabsl(z))) break;
  }

  return z;
}

bool _polynomial_find_roots_companion(polynomial_t *poly,
                                      cxldouble *roots,
                                      size_t *num_roots) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0 || !roots || !num_roots) {
    return false;
  }

  size_t n = poly->degree;
  *num_roots = 0;

  // Handle degree 1 directly: ax + b = 0 => x = -b/a
  if (n == 1) {
    roots[0] = cxdivl(cxnegl(poly->coeffs[0]), poly->coeffs[1]);
    *num_roots = 1;
    return true;
  }

  // Handle degree 2 with quadratic formula for better precision
  if (n == 2) {
    cxldouble a = poly->coeffs[2];
    cxldouble b = poly->coeffs[1];
    cxldouble c = poly->coeffs[0];
    // disc = sqrt(b*b - 4*a*c)
    cxldouble b2 = cxmull(b, b);
    cxldouble ac4 = cxscalel(cxmull(a, c), 4.0L);
    cxldouble disc = cxsqrtl(cxsubl(b2, ac4));
    // roots = (-b ± disc) / (2*a)
    cxldouble neg_b = cxnegl(b);
    cxldouble denom = cxscalel(a, 2.0L);
    roots[0] = cxdivl(cxaddl(neg_b, disc), denom);
    roots[1] = cxdivl(cxsubl(neg_b, disc), denom);
    *num_roots = 2;
    return true;
  }

  // Build companion matrix
  cxdouble *matrix = build_companion_matrix(poly, n);
  if (!matrix) return false;

  // Allocate eigenvalue array
  cxdouble *eigenvalues = malloc(n * sizeof(cxdouble));
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
    cxldouble raw_root = cx_to_cxl(eigenvalues[i]);
    roots[i] = polish_root_companion(poly, raw_root);
  }

  free(eigenvalues);
  *num_roots = n;
  return true;
}

// Compare function for sorting complex numbers
static int complex_compare_d(const void *a, const void *b) {
  cxldouble ca = *(const cxldouble *) a;
  cxldouble cb = *(const cxldouble *) b;

  long double real_diff = cxreall(ca) - cxreall(cb);
  if (fabsl(real_diff) > LDBL_EPSILON) {
    return (real_diff > 0) - (real_diff < 0);
  }

  long double imag_diff = cximagl(ca) - cximagl(cb);
  return (imag_diff > 0) - (imag_diff < 0);
}

bool polynomial_find_roots_companion(polynomial_t *poly) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0) {
    return false;
  }

  cxldouble *raw_roots = malloc(poly->degree * sizeof(cxldouble));
  if (!raw_roots) return false;

  size_t num_roots;
  bool success = _polynomial_find_roots_companion(poly, raw_roots, &num_roots);

  if (!success || num_roots == 0) {
    free(raw_roots);
    return false;
  }

  // Sort roots for deduplication
  qsort(raw_roots, num_roots, sizeof(cxldouble), complex_compare_d);

  // Deduplicate and compute multiplicities
  size_t distinct_count = 0;
  cxldouble prev_root = CXL(0.0, 0.0);

  for (size_t i = 0; i < num_roots; i++) {
    if (distinct_count == 0 ||
        !complex_approx_equal_d(cxl_to_cx(raw_roots[i]), cxl_to_cx(prev_root))) {
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