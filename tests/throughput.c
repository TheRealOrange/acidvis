//
// Created by Lin Yicheng on 1/12/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "compat_complex.h"

#include "polynomial.h"
#include "util.h"

#define NUM_BASE 3

static double get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
  // generates polynomials
  size_t degree = 12;
  size_t num_base = NUM_BASE;
  size_t num_combinations = int_pow(NUM_BASE, degree+1);

  printf("batched lapack optimization example\n");

  // base coefficients
  cxldouble base_coeffs[NUM_BASE] = {-1.0, 1.5, 1.0};

  // allocate result arrays
  cxldouble *roots = malloc(num_combinations * degree * sizeof(cxldouble));
  size_t *num_distinct = malloc(num_combinations * sizeof(size_t));

  if (!roots || !num_distinct) {
    printf("memory allocation failed\n");
    if (roots) free(roots);
    if (num_distinct) free(num_distinct);
    return 1;
  }

  printf("configuration:\n");
  printf("  polynomial degree: %zu\n", degree);
  printf("  base coefficients: %zu\n", num_base);
  printf("  total combinations: %zu\n", num_combinations);
  printf("  threads: %d\n\n", get_num_threads());

#ifdef HAVE_LAPACK
  printf("testing batched lapack implementation...\n");

  // run it a few times to get stable timing
  double total_time = 0;
  int runs = 3;

  for (int run = 0; run < runs; run++) {
    double start = get_time();

    size_t total_roots = polynomial_find_root_combinations_companion(
      base_coeffs, num_base, degree,
      &roots, &num_distinct, num_combinations
    );

    double elapsed = get_time() - start;
    total_time += elapsed;

    printf("  run %d: %.3f seconds (%zu roots found)\n",
           run + 1, elapsed, total_roots);
  }

  double avg_time = total_time / runs;
  double polys_per_sec = num_combinations / avg_time;
  double roots_per_sec = (num_combinations * degree) / avg_time;

  printf("\nresults:\n");
  printf("  average time: %.3f seconds\n", avg_time);
  printf("  throughput: %.0f polynomials/second\n", polys_per_sec);
  printf("  throughput: %.0f roots/second\n", roots_per_sec);

  // check a sample result
  printf("\nsample results (first 3 polynomials):\n");
  for (size_t i = 0; i < 3 && i < num_combinations; i++) {
    printf("  polynomial %zu: found %zu roots\n", i, num_distinct[i]);
    if (num_distinct[i] > 0) {
      printf("    first root: %.6Lf + %.6Lfi\n",
             cxreall(roots[i * degree]),
             cximagl(roots[i * degree]));
    }
  }

  printf("\nbatch details:\n");
  printf("  memory per polynomial: ~%zu bytes\n",
         degree * degree * 16 + degree * 16);

  size_t bytes_per_poly = degree * degree * 16 + degree * 16;
  size_t batch_size = (MAX_BATCH_MEM_MB * 1024 * 1024) / bytes_per_poly;
  if (batch_size > MAX_BATCH_SIZE) batch_size = MAX_BATCH_SIZE;
  if (batch_size > num_combinations) batch_size = num_combinations;

  printf("  batch size used: ~%zu polynomials\n", batch_size);
  printf("  number of batches: ~%zu\n",
         (num_combinations + batch_size - 1) / batch_size);
  printf("  memory per batch: ~%.1f MB\n",
         (double) (batch_size * bytes_per_poly) / (1024 * 1024));

#else
  printf("lapack not available - using jenkins-traub\n");
  printf("(batched optimization only available with lapack)\n\n");

  double start = get_time();

  size_t total_roots = polynomial_find_root_combinations(
    base_coeffs, num_base, degree,
    &roots, &num_distinct, num_combinations
  );

  double elapsed = get_time() - start;

  printf("completed in %.3f seconds (%zu roots found)\n", elapsed, total_roots);
  printf("throughput: %.0f polynomials/second\n", num_combinations / elapsed);
#endif

  free(roots);
  free(num_distinct);

  printf("\n");
  return 0;
}
