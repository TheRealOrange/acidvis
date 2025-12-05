//
// Created by Lin Yicheng on 1/12/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "compat_complex.h"

#include "polynomial.h"
#include "util.h"

#ifdef HAVE_LAPACK
#include "companion.h"
#endif

static double get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// create a test polynomial of given degree with roots on unit circle
static polynomial_t *create_test_polynomial(size_t degree) {
  cxldouble *roots = calloc(degree, sizeof(cxldouble));
  for (size_t i = 0; i < degree; i++) {
    long double angle = ((long double) i * M_PI * 2.0L) / (long double) degree;
    roots[i] = cxexpl(I * angle);
  }
  polynomial_t *p = polynomial_from_roots(roots, degree, false);
  free(roots);
  if (p) {
    polynomial_find_roots(p);
  }
  return p;
}

static void benchmark_combinations(size_t degree, bool use_lapack) {
  printf("\n=== benchmark: degree %zu, %s ===\n",
         degree, use_lapack ? "LAPACK (companion)" : "Jenkins-Traub");

  // generates polynomials
  size_t num_coeffs = 3;
  size_t num_combinations = int_pow(3, degree+1);

  printf("batched lapack optimization example\n");

  // base coefficients
  cxldouble base_coeffs[3] = {-1.0, 1.5, 1.0};

  printf("number of combinations: %zu\n", num_combinations);
  printf("number of threads: %d\n", get_num_threads());

  // Allocate result arrays
  cxldouble *roots = malloc(num_combinations * degree * sizeof(cxldouble));
  size_t *num_distinct = malloc(num_combinations * sizeof(size_t));

  if (!roots || !num_distinct) {
    printf("Failed to allocate memory\n");
    return;
  }

  // warm-up run
  printf("warm-up run...\n");
#ifdef HAVE_LAPACK
  if (use_lapack) {
    polynomial_find_root_combinations_companion(base_coeffs, num_coeffs, degree,
                                                &roots, &num_distinct, num_combinations);
  } else {
    polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                      &roots, &num_distinct, num_combinations);
  }
#else
  (void) use_lapack;
  polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                    &roots, &num_distinct, num_combinations);
#endif

  int num_runs = 3;
  double total_time = 0;
  size_t total_roots = 0;

  printf("timing %d runs...\n", num_runs);

  for (int run = 0; run < num_runs; run++) {
    double start = get_time();

#ifdef HAVE_LAPACK
    if (use_lapack) {
      polynomial_find_root_combinations_companion(base_coeffs, num_coeffs, degree,
                                                  &roots, &num_distinct, num_combinations);
    } else {
      polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                        &roots, &num_distinct, num_combinations);
    }
#else
    (void) use_lapack;
    polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                      &roots, &num_distinct, num_combinations);
#endif

    double end = get_time();
    double elapsed = end - start;
    total_time += elapsed;

    printf("  run %d: %.3f seconds (%zu roots found)\n", run + 1, elapsed, total_roots);
  }

  double avg_time = total_time / num_runs;
  double combs_per_sec = num_combinations / avg_time;

  printf("\nresults:\n");
  printf("  average time: %.3f seconds\n", avg_time);
  printf("  throughput: %.1f combinations/second\n", combs_per_sec);
  printf("  total roots found: %zu\n", total_roots);

  free(roots);
  free(num_distinct);
}

static void benchmark_scaling(size_t degree, bool use_lapack) {
  printf("\n=== thread scaling test: degree %zu ===\n", degree);

  // generates polynomials
  size_t num_coeffs = 3;
  size_t num_combinations = int_pow(3, degree+1);

  printf("batched lapack optimization example\n");

  // base coefficients
  cxldouble base_coeffs[3] = {-1.0, 1.5, 1.0};

  cxldouble *roots = malloc(num_combinations * degree * sizeof(cxldouble));
  size_t *num_distinct = malloc(num_combinations * sizeof(size_t));

  if (!roots || !num_distinct) {
    printf("failed to allocate memory\n");
    return;
  }

  int max_threads = get_num_threads();
  printf("max threads available: %d\n", max_threads);
  printf("number of combinations: %zu\n\n", num_combinations);

  double baseline_time = 0;

  // Test with different thread counts
  for (int threads = 1; threads <= max_threads; threads *= 2) {
    set_num_threads(threads);

    // Warm-up
#ifdef HAVE_LAPACK
    if (use_lapack) {
      polynomial_find_root_combinations_companion(base_coeffs, num_coeffs, degree,
                                                  &roots, &num_distinct, num_combinations);
    } else {
      polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                        &roots, &num_distinct, num_combinations);
    }
#else
    polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                      &roots, &num_distinct, num_combinations);
#endif

    // Timed run
    double start = get_time();

#ifdef HAVE_LAPACK
    if (use_lapack) {
      polynomial_find_root_combinations_companion(base_coeffs, num_coeffs, degree,
                                                  &roots, &num_distinct, num_combinations);
    } else {
      polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                        &roots, &num_distinct, num_combinations);
    }
#else
    polynomial_find_root_combinations(base_coeffs, num_coeffs, degree,
                                      &roots, &num_distinct, num_combinations);
#endif

    double elapsed = get_time() - start;

    if (threads == 1) {
      baseline_time = elapsed;
    }

    double speedup = baseline_time / elapsed;
    double efficiency = speedup / threads * 100.0;

    printf("threads: %2d | time: %.3fs | speedup: %.2fx | efficiency: %.1f%%\n",
           threads, elapsed, speedup, efficiency);
  }

  // Restore max threads
  set_num_threads(max_threads);

  free(roots);
  free(num_distinct);
}

int main(int argc, char *argv[]) {
  printf("OpenMP threads available: %d\n", get_num_threads());

#ifdef HAVE_LAPACK
  printf("LAPACK: enabled\n");
#else
  printf("LAPACK: disabled\n");
#endif

  // Default benchmark parameters
  size_t test_degree = 8;
  bool use_lapack = false;

  // Parse command line
  if (argc > 1) {
    test_degree = (size_t) atoi(argv[1]);
    if (test_degree < 2) test_degree = 2;
    if (test_degree > 8) {
      printf("warning: degree %zu will have %zu combinations, may take a long time\n",
             test_degree, int_pow(3, test_degree+1));
    }
  }

  if (argc > 2) {
    use_lapack = (argv[2][0] == '1' || argv[2][0] == 'y' || argv[2][0] == 'Y');
  }

  // Run benchmarks
  benchmark_combinations(test_degree, use_lapack);

  // Thread scaling test
  if (get_num_threads() > 1) {
    benchmark_scaling(test_degree, use_lapack);
  }

  // Also test with LAPACK if available and not already tested
#ifdef HAVE_LAPACK
  if (!use_lapack) {
    benchmark_combinations(test_degree, true);
    if (get_num_threads() > 1) {
      benchmark_scaling(test_degree, true);
    }
  }
#endif

  printf("\nbenchmark complete.\n");
  return 0;
}
