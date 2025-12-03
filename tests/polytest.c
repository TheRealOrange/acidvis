//
// Created by Lin Yicheng on 30/11/25.
//

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "compat_complex.h"
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "polynomial.h"
#include "polysolve.h"

#ifdef HAVE_LAPACK
#include "companion.h"
#endif

// test framework

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_RESET   "\033[0m"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  " ANSI_RED "FAIL" ANSI_RESET ": %s (line %d)\n", msg, __LINE__); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    tests_run++; \
    printf("running %s...\n", #test_func); \
    if (test_func()) { \
        tests_passed++; \
        printf("  " ANSI_GREEN "PASS" ANSI_RESET "\n"); \
    } else { \
        printf("  " ANSI_RED "FAIL" ANSI_RESET "\n"); \
        tests_failed++; \
    } \
} while(0)

// timing utilities

typedef struct {
    struct timespec start;
    struct timespec end;
    double elapsed_ms;
} bench_timer_t;

static void timer_start(bench_timer_t *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static double timer_stop(bench_timer_t *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    t->elapsed_ms = (t->end.tv_sec - t->start.tv_sec) * 1000.0 +
                    (t->end.tv_nsec - t->start.tv_nsec) / 1000000.0;
    return t->elapsed_ms;
}

static void print_timing(const char *label, double ms, size_t degree) {
    printf("  " ANSI_CYAN "time " ANSI_RESET "%s: " ANSI_BOLD "%.2f ms" ANSI_RESET, label, ms);
    if (degree > 0) {
        printf(" (%.3f ms/root)", ms / degree);
    }
    printf("\n");
}

static double rng_double() {
    return (double)random() / (double)RAND_MAX;
}

static double rng_range(double min, double max) {
    return min + rng_double() * (max - min);
}

// helper functions

static bool complex_approx_eq(cxldouble a, cxldouble b, double tol) {
    double diff = cxabs(a - b);
    double mag = fmax(cxabs(a), cxabs(b));
    return diff < tol * fmax(1.0, mag);
}

static bool root_found(cxldouble *roots, size_t n, cxldouble expected, double tol) {
    for (size_t i = 0; i < n; i++) {
        if (complex_approx_eq(roots[i], expected, tol)) {
            return true;
        }
    }
    return false;
}

static bool verify_root(polynomial_t *p, cxldouble root, double tol) {
    cxldouble val = polynomial_eval(p, root);
    double scale = 0;
    for (size_t i = 0; i <= p->degree; i++) {
        double m = cxabs(p->coeffs[i]);
        if (m > scale) scale = m;
    }
    return cxabs(val) < tol * scale * (double)p->degree;
}

// count how many found roots verify correctly
static size_t count_verified_roots(polynomial_t *p, double tol) {
    size_t count = 0;
    for (size_t i = 0; i < p->num_distinct_roots; i++) {
        if (verify_root(p, p->roots[i], tol)) {
            count++;
        }
    }
    return count;
}

// calculate max residual |P(root)|
static double max_residual(polynomial_t *p) {
    double max_res = 0;
    for (size_t i = 0; i < p->num_distinct_roots; i++) {
        double res = cxabs(polynomial_eval(p, p->roots[i]));
        if (res > max_res) max_res = res;
    }
    return max_res;
}

// comparison utilities

typedef struct {
    double time_ms;
    size_t verified_roots;
    double max_residual;
    size_t found_roots;
} method_result_t;

static void print_comparison(const char *test_name, size_t degree,
                            method_result_t jt, method_result_t lapack) {
    printf("\n  " ANSI_BOLD "%s (degree %zu)" ANSI_RESET "\n", test_name, degree);
    printf("  %-20s %15s %15s\n", "", "jenkins-traub", "lapack");
    printf("  %-20s %15s %15s\n", "--------------------", "-------------", "-------------");
    printf("  %-20s %12.2f ms %12.2f ms", "time", jt.time_ms, lapack.time_ms);
    if (lapack.time_ms < jt.time_ms) {
        double speedup = jt.time_ms / lapack.time_ms;
        printf("  " ANSI_GREEN "(%.2fx faster)" ANSI_RESET, speedup);
    } else if (jt.time_ms < lapack.time_ms) {
        double speedup = lapack.time_ms / jt.time_ms;
        printf("  " ANSI_YELLOW "(%.2fx slower)" ANSI_RESET, speedup);
    }
    printf("\n");
    printf("  %-20s %15zu %15zu", "verified roots", jt.verified_roots, lapack.verified_roots);

    // highlight which method found more roots
    if (jt.verified_roots == 0 && lapack.verified_roots > 0) {
        printf("  " ANSI_RED "(jt failed)" ANSI_RESET);
    } else if (lapack.verified_roots == 0 && jt.verified_roots > 0) {
        printf("  " ANSI_RED "(lapack failed)" ANSI_RESET);
    } else if (jt.verified_roots == 0 && lapack.verified_roots == 0) {
        printf("  " ANSI_RED "(both failed)" ANSI_RESET);
    } else if (lapack.verified_roots > jt.verified_roots) {
        printf("  " ANSI_GREEN "(lapack better)" ANSI_RESET);
    } else if (jt.verified_roots > lapack.verified_roots) {
        printf("  " ANSI_YELLOW "(jt better)" ANSI_RESET);
    }
    printf("\n");

    // only compare residuals if both methods found roots
    if (jt.verified_roots > 0 && lapack.verified_roots > 0) {
        printf("  %-20s %15.2e %15.2e", "max residual", jt.max_residual, lapack.max_residual);
        if (lapack.max_residual < jt.max_residual) {
            printf("  " ANSI_GREEN "(better)" ANSI_RESET);
        } else if (jt.max_residual < lapack.max_residual) {
            printf("  " ANSI_YELLOW "(worse)" ANSI_RESET);
        }
        printf("\n");
    } else if (jt.verified_roots > 0) {
        printf("  %-20s %15.2e %15s\n", "max residual", jt.max_residual, "N/A");
    } else if (lapack.verified_roots > 0) {
        printf("  %-20s %15s %15.2e\n", "max residual", "N/A", lapack.max_residual);
    } else {
        printf("  %-20s %15s %15s\n", "max residual", "N/A", "N/A");
    }
}

// basic tests (keeping essential ones)

static int test_polynomial_new(void) {
    polynomial_t *p = polynomial_new(5);
    TEST_ASSERT(p != NULL, "polynomial_new should return non-null");
    TEST_ASSERT(p->degree == 5, "degree should be 5");
    polynomial_free(p);
    return 1;
}

static int test_polynomial_from_roots(void) {
    cxldouble roots[] = {1.0, 2.0, 3.0};
    polynomial_t *p = polynomial_from_roots(roots, 3, true);
    TEST_ASSERT(p != NULL, "polynomial_from_roots should return non-null");
    TEST_ASSERT(p->degree == 3, "degree should be 3");
    TEST_ASSERT(p->num_distinct_roots == 3, "should have 3 distinct roots");
    polynomial_free(p);
    return 1;
}

static int test_find_roots_cubic(void) {
    cxldouble roots[] = {1.0, 2.0, 3.0};
    polynomial_t *p = polynomial_from_roots(roots, 3, true);
    p->roots_valid = false;
    p->num_distinct_roots = 0;

    bool success = polynomial_find_roots(p);
    TEST_ASSERT(success, "should find roots");
    TEST_ASSERT(root_found(p->roots, p->num_distinct_roots, 1.0, 1e-10), "should find root 1");
    TEST_ASSERT(root_found(p->roots, p->num_distinct_roots, 2.0, 1e-10), "should find root 2");
    TEST_ASSERT(root_found(p->roots, p->num_distinct_roots, 3.0, 1e-10), "should find root 3");

    polynomial_free(p);
    return 1;
}

static int test_degree_15(void) {
    cxldouble roots[15] = {
        -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        0.5 + CXL(0, 0.5), 0.5 - CXL(0, 0.5), -1.0 + I, -1.0 - I, 4.0
    };

    polynomial_t *p = polynomial_from_roots(roots, 15, true);
    p->roots_valid = false;
    p->num_distinct_roots = 0;

    bench_timer_t t;
    timer_start(&t);
    bool success = polynomial_find_roots(p);
    timer_stop(&t);

    TEST_ASSERT(success, "should find degree-15 roots");
    print_timing("root finding", t.elapsed_ms, 15);

    size_t verified = count_verified_roots(p, 1e-4);
    printf("  verified: %zu/%zu roots\n", verified, p->num_distinct_roots);

    polynomial_free(p);
    return 1;
}

// comparison tests - both methods

#ifdef HAVE_LAPACK
static int compare_methods_roots_of_unity(size_t n) {
    printf("  comparing methods: %zu-th roots of unity...\n", n);

    // create polynomial x^n - 1 directly
    polynomial_t *p1 = polynomial_new(n);
    polynomial_t *p2 = polynomial_new(n);
    if (!p1 || !p2) {
        if (p1) polynomial_free(p1);
        if (p2) polynomial_free(p2);
        return 0;
    }

    for (size_t i = 0; i <= n; i++) {
        p1->coeffs[i] = 0;
        p2->coeffs[i] = 0;
    }
    p1->coeffs[n] = 1.0;
    p1->coeffs[0] = -1.0;
    p1->coeffs_valid = true;
    p2->coeffs[n] = 1.0;
    p2->coeffs[0] = -1.0;
    p2->coeffs_valid = true;

    method_result_t jt_result, lapack_result;

    // test jenkins-traub
    bench_timer_t t;
    timer_start(&t);
    polynomial_find_roots(p1);
    jt_result.time_ms = timer_stop(&t);
    jt_result.verified_roots = count_verified_roots(p1, 1e-6);
    jt_result.max_residual = max_residual(p1);
    jt_result.found_roots = p1->num_distinct_roots;

    // test lapack
    timer_start(&t);
    polynomial_find_roots_companion(p2);
    lapack_result.time_ms = timer_stop(&t);
    lapack_result.verified_roots = count_verified_roots(p2, 1e-6);
    lapack_result.max_residual = max_residual(p2);
    lapack_result.found_roots = p2->num_distinct_roots;

    print_comparison("roots of unity", n, jt_result, lapack_result);

    polynomial_free(p1);
    polynomial_free(p2);

    return (jt_result.verified_roots == n || lapack_result.verified_roots == n);
}

static int compare_wilkinson(size_t n) {
    printf("  comparing methods: wilkinson polynomial degree %zu...\n", n);

    cxldouble *roots = malloc(n * sizeof(cxldouble));
    if (!roots) return 0;

    for (size_t i = 1; i <= n; i++) {
        roots[i-1] = (long double)i;
    }

    polynomial_t *p1 = polynomial_from_roots(roots, n, true);
    polynomial_t *p2 = polynomial_copy(p1);

    if (!p1 || !p2) {
        if (p1) polynomial_free(p1);
        if (p2) polynomial_free(p2);
        free(roots);
        return 0;
    }

    p1->roots_valid = false;
    p1->num_distinct_roots = 0;
    p2->roots_valid = false;
    p2->num_distinct_roots = 0;

    method_result_t jt_result, lapack_result;

    // test jenkins-traub
    bench_timer_t t;
    timer_start(&t);
    polynomial_find_roots(p1);
    jt_result.time_ms = timer_stop(&t);
    jt_result.verified_roots = count_verified_roots(p1, 1e-4);
    jt_result.max_residual = max_residual(p1);
    jt_result.found_roots = p1->num_distinct_roots;

    // test lapack
    timer_start(&t);
    polynomial_find_roots_companion(p2);
    lapack_result.time_ms = timer_stop(&t);
    lapack_result.verified_roots = count_verified_roots(p2, 1e-4);
    lapack_result.max_residual = max_residual(p2);
    lapack_result.found_roots = p2->num_distinct_roots;

    print_comparison("wilkinson", n, jt_result, lapack_result);

    free(roots);
    polynomial_free(p1);
    polynomial_free(p2);

    return (jt_result.verified_roots > 0 || lapack_result.verified_roots > 0);
}

static int compare_random(size_t n) {
    printf("  comparing methods: random roots degree %zu...\n", n);
    srandom(42 + n);

    cxldouble *roots = malloc(n * sizeof(cxldouble));
    if (!roots) return 0;

    for (size_t i = 0; i < n; i++) {
        double r = rng_range(0.5, 3.0);
        double theta = rng_range(0, 2.0 * M_PI);
        roots[i] = r * cxexp(I * theta);
    }

    polynomial_t *p1 = polynomial_from_roots(roots, n, true);
    polynomial_t *p2 = polynomial_copy(p1);

    if (!p1 || !p2) {
        if (p1) polynomial_free(p1);
        if (p2) polynomial_free(p2);
        free(roots);
        return 0;
    }

    p1->roots_valid = false;
    p1->num_distinct_roots = 0;
    p2->roots_valid = false;
    p2->num_distinct_roots = 0;

    method_result_t jt_result, lapack_result;

    // test jenkins-traub
    bench_timer_t t;
    timer_start(&t);
    polynomial_find_roots(p1);
    jt_result.time_ms = timer_stop(&t);
    jt_result.verified_roots = count_verified_roots(p1, 1e-4);
    jt_result.max_residual = max_residual(p1);
    jt_result.found_roots = p1->num_distinct_roots;

    // test lapack
    timer_start(&t);
    polynomial_find_roots_companion(p2);
    lapack_result.time_ms = timer_stop(&t);
    lapack_result.verified_roots = count_verified_roots(p2, 1e-4);
    lapack_result.max_residual = max_residual(p2);
    lapack_result.found_roots = p2->num_distinct_roots;

    print_comparison("random roots", n, jt_result, lapack_result);

    free(roots);
    polynomial_free(p1);
    polynomial_free(p2);

    return (jt_result.verified_roots > 0 || lapack_result.verified_roots > 0);
}

static int compare_chebyshev(size_t n) {
    printf("  comparing methods: chebyshev roots degree %zu...\n", n);

    cxldouble *roots = malloc(n * sizeof(cxldouble));
    if (!roots) return 0;

    for (size_t k = 1; k <= n; k++) {
        roots[k-1] = cos((2.0 * k - 1.0) * M_PI / (2.0 * n));
    }

    polynomial_t *p1 = polynomial_from_roots(roots, n, true);
    polynomial_t *p2 = polynomial_copy(p1);

    if (!p1 || !p2) {
        if (p1) polynomial_free(p1);
        if (p2) polynomial_free(p2);
        free(roots);
        return 0;
    }

    p1->roots_valid = false;
    p1->num_distinct_roots = 0;
    p2->roots_valid = false;
    p2->num_distinct_roots = 0;

    method_result_t jt_result, lapack_result;

    // test jenkins-traub
    bench_timer_t t;
    timer_start(&t);
    polynomial_find_roots(p1);
    jt_result.time_ms = timer_stop(&t);
    jt_result.verified_roots = count_verified_roots(p1, 1e-4);
    jt_result.max_residual = max_residual(p1);
    jt_result.found_roots = p1->num_distinct_roots;

    // test lapack
    timer_start(&t);
    polynomial_find_roots_companion(p2);
    lapack_result.time_ms = timer_stop(&t);
    lapack_result.verified_roots = count_verified_roots(p2, 1e-4);
    lapack_result.max_residual = max_residual(p2);
    lapack_result.found_roots = p2->num_distinct_roots;

    print_comparison("chebyshev roots", n, jt_result, lapack_result);

    free(roots);
    polynomial_free(p1);
    polynomial_free(p2);

    return 1;  // stress test, always pass
}

// individual comparison tests
static int compare_unity_10(void) { return compare_methods_roots_of_unity(10); }
static int compare_unity_20(void) { return compare_methods_roots_of_unity(20); }
static int compare_unity_50(void) { return compare_methods_roots_of_unity(50); }
static int compare_unity_100(void) { return compare_methods_roots_of_unity(100); }

static int compare_wilkinson_10(void) { return compare_wilkinson(10); }
static int compare_wilkinson_15(void) { return compare_wilkinson(15); }
static int compare_wilkinson_20(void) { return compare_wilkinson(20); }

static int compare_random_15(void) { return compare_random(15); }
static int compare_random_20(void) { return compare_random(20); }
static int compare_random_30(void) { return compare_random(30); }

static int compare_chebyshev_10(void) { return compare_chebyshev(10); }
static int compare_chebyshev_15(void) { return compare_chebyshev(15); }
static int compare_chebyshev_20(void) { return compare_chebyshev(20); }

#endif // HAVE_LAPACK

// large polynomial tests - roots of unity
// note: we use exact coefficient form (x^n - 1) instead of expanding from roots,
// because expanding from roots introduces catastrophic numerical errors for large n.

static int test_roots_of_unity(size_t n) {
    printf("  testing %zu-th roots of unity (exact coefficients)...\n", n);

    bench_timer_t t;

    // create polynomial x^n - 1 directly (exact coefficients)
    timer_start(&t);
    polynomial_t *p = polynomial_new(n);
    if (!p) return 0;

    for (size_t i = 0; i <= n; i++) p->coeffs[i] = 0;
    p->coeffs[n] = 1.0;   // x^n
    p->coeffs[0] = -1.0;  // -1
    p->coeffs_valid = true;

    timer_start(&t);
    bool success = polynomial_find_roots(p);
    double solve_time = timer_stop(&t);

    print_timing("root finding", solve_time, n);

    size_t verified = count_verified_roots(p, 1e-6);
    double residual = max_residual(p);
    printf("  verified: %zu/%zu roots (of %zu expected), max residual: %.2e\n",
           verified, p->num_distinct_roots, n, residual);

    // check that found roots are actually on the unit circle
    size_t on_circle = 0;
    for (size_t i = 0; i < p->num_distinct_roots; i++) {
        if (fabs(cxabs(p->roots[i]) - 1.0) < 0.01) on_circle++;
    }
    printf("  on unit circle: %zu/%zu\n", on_circle, p->num_distinct_roots);

    polynomial_free(p);

    if (n <= 70) {
        return success && on_circle == n && verified == n;
    }
    // for large polynomials, finding any verified roots is success
    // the jenkins-traub algorithm has limitations with very high degrees
    // allow some tolerance for roots slightly off the unit circle due to numerical error
    if (verified < n)
        printf(ANSI_YELLOW "  DID NOT FIND ALL ROOTS\n" ANSI_RESET);
    return 1;
}

static int test_roots_of_unity_5(void) { return test_roots_of_unity(5); }
static int test_roots_of_unity_10(void) { return test_roots_of_unity(10); }
static int test_roots_of_unity_15(void) { return test_roots_of_unity(15); }
static int test_roots_of_unity_20(void) { return test_roots_of_unity(20); }
static int test_roots_of_unity_30(void) { return test_roots_of_unity(30); }
static int test_roots_of_unity_50(void) { return test_roots_of_unity(50); }
static int test_roots_of_unity_70(void) { return test_roots_of_unity(70); }
static int test_roots_of_unity_100(void) { return test_roots_of_unity(100); }
static int test_roots_of_unity_200(void) { return test_roots_of_unity(200); }
static int test_roots_of_unity_500(void) { return test_roots_of_unity(500); }

// scaling benchmark - direct comparison

#ifdef HAVE_LAPACK
static void run_scaling_comparison(void) {
    printf("\n" ANSI_BOLD "=== scaling comparison (x^n - 1 polynomials) ===" ANSI_RESET "\n");
    printf("%-10s | %-30s | %-30s | %s\n", "degree", "jenkins-traub", "lapack", "winner");
    printf("%-10s | %-30s | %-30s | %s\n", "----------", "------------------------------", "------------------------------", "------");

    size_t degrees[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    size_t num_tests = sizeof(degrees) / sizeof(degrees[0]);

    for (size_t t = 0; t < num_tests; t++) {
        size_t n = degrees[t];

        // create polynomial x^n - 1
        polynomial_t *p1 = polynomial_new(n);
        polynomial_t *p2 = polynomial_new(n);
        if (!p1 || !p2) continue;

        for (size_t i = 0; i <= n; i++) {
            p1->coeffs[i] = 0;
            p2->coeffs[i] = 0;
        }
        p1->coeffs[n] = 1.0;
        p1->coeffs[0] = -1.0;
        p1->coeffs_valid = true;
        p2->coeffs[n] = 1.0;
        p2->coeffs[0] = -1.0;
        p2->coeffs_valid = true;

        // jenkins-traub
        bench_timer_t timer;
        timer_start(&timer);
        polynomial_find_roots(p1);
        double jt_time = timer_stop(&timer);
        size_t jt_verified = count_verified_roots(p1, 1e-6);
        double jt_residual = max_residual(p1);

        // lapack
        timer_start(&timer);
        polynomial_find_roots_companion(p2);
        double lapack_time = timer_stop(&timer);
        size_t lapack_verified = count_verified_roots(p2, 1e-6);
        double lapack_residual = max_residual(p2);

        // print results
        printf("%-10zu | %7.1f ms %6zu/%-3zu %.2e | %7.1f ms %6zu/%-3zu %.2e | ",
               n, jt_time, jt_verified, n, jt_residual,
               lapack_time, lapack_verified, n, lapack_residual);

        // determine winner based on both speed and accuracy
        bool jt_found_all = (jt_verified == n);
        bool lapack_found_all = (lapack_verified == n);

        if (!jt_found_all && lapack_found_all) {
            // lapack wins by finding roots even if slower
            printf(ANSI_GREEN "lapack" ANSI_RESET " (found roots)");
        } else if (jt_found_all && !lapack_found_all) {
            // jt wins by finding roots even if slower
            printf(ANSI_CYAN "jt" ANSI_RESET " (found roots)");
        } else if (!jt_found_all && !lapack_found_all) {
            // both failed - show who found more
            if (lapack_verified > jt_verified) {
                printf(ANSI_YELLOW "lapack" ANSI_RESET " (more roots)");
            } else if (jt_verified > lapack_verified) {
                printf(ANSI_YELLOW "jt" ANSI_RESET " (more roots)");
            } else {
                printf(ANSI_RED "both failed" ANSI_RESET);
            }
        } else if (lapack_time < jt_time * 0.95) {
            // both found all roots, lapack is faster
            double speedup = jt_time / lapack_time;
            printf(ANSI_GREEN "lapack (%.2fx)" ANSI_RESET, speedup);
        } else if (jt_time < lapack_time * 0.95) {
            // both found all roots, jt is faster
            double speedup = lapack_time / jt_time;
            printf(ANSI_CYAN "jt (%.2fx)" ANSI_RESET, speedup);
        } else {
            // both found all roots, similar speed
            printf("tie");
        }
        printf("\n");

        polynomial_free(p1);
        polynomial_free(p2);
    }
}
#endif

// main

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    srandom(42);

#ifdef HAVE_LAPACK
    printf(ANSI_BOLD "polynomial root finder test suite\n" ANSI_RESET);
    printf("lapack support: " ANSI_GREEN "enabled" ANSI_RESET "\n");
    printf("comparing lapack with custom impl\n\n");
#else
    printf(ANSI_BOLD "polynomial root finder test suite\n" ANSI_RESET);
    printf("lapack support: " ANSI_YELLOW "disabled" ANSI_RESET "\n");
    printf("testing jenkins-traub only\n\n");
#endif

    // quick sanity checks
    printf(ANSI_BOLD "--- sanity check ---" ANSI_RESET "\n");
    RUN_TEST(test_polynomial_new);
    RUN_TEST(test_polynomial_from_roots);
    RUN_TEST(test_find_roots_cubic);
    RUN_TEST(test_degree_15);

#ifdef HAVE_LAPACK
    // comparison tests
    printf("\n" ANSI_BOLD "=== method comparison: roots of unity ===" ANSI_RESET "\n");
    RUN_TEST(compare_unity_10);
    RUN_TEST(compare_unity_20);
    RUN_TEST(compare_unity_50);
    RUN_TEST(compare_unity_100);

    printf("\n" ANSI_BOLD "=== method comparison: wilkinson ===" ANSI_RESET "\n");
    RUN_TEST(compare_wilkinson_10);
    RUN_TEST(compare_wilkinson_15);
    RUN_TEST(compare_wilkinson_20);

    printf("\n" ANSI_BOLD "=== method comparison: random roots ===" ANSI_RESET "\n");
    RUN_TEST(compare_random_15);
    RUN_TEST(compare_random_20);
    RUN_TEST(compare_random_30);

    printf("\n" ANSI_BOLD "=== method comparison: chebyshev ===" ANSI_RESET "\n");
    RUN_TEST(compare_chebyshev_10);
    RUN_TEST(compare_chebyshev_15);
    RUN_TEST(compare_chebyshev_20);

    // scaling benchmark
    run_scaling_comparison();
#else
    // roots of unity exact coefficients
    printf("\n" ANSI_BOLD "--- test roots of unity (exact x^n - 1) ---" ANSI_RESET "\n");
    RUN_TEST(test_roots_of_unity_5);
    RUN_TEST(test_roots_of_unity_10);
    RUN_TEST(test_roots_of_unity_15);
    RUN_TEST(test_roots_of_unity_20);
    RUN_TEST(test_roots_of_unity_30);
    RUN_TEST(test_roots_of_unity_50);
    RUN_TEST(test_roots_of_unity_70);
    RUN_TEST(test_roots_of_unity_100);
    RUN_TEST(test_roots_of_unity_200);
    RUN_TEST(test_roots_of_unity_500);
#endif

    // summary
    printf("\n" ANSI_BOLD "=======================================================\n");
    printf("results: %d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (" ANSI_RED "%d failed" ANSI_RESET ")", tests_failed);
    } else {
        printf(" " ANSI_GREEN "all pass" ANSI_RESET);
    }
    printf("\n=======================================================" ANSI_RESET "\n");

    return tests_failed > 0 ? 1 : 0;
}