//
// Created by Lin Yicheng on 30/11/25.
//

#include "polynomial.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lapacksolve.h"
#include "polysolve.h"
#include "util.h"

#define REL_EPSILON 1e-20

static bool complex_approx_equal(cxldouble a, cxldouble b) {
  long double diff = cxabsl(cxsubl(a, b));
  long double mag = fmaxl(cxabsl(a), cxabsl(b));
  return diff < (LDBL_EPSILON + REL_EPSILON * mag);
}

// need consistent ordering, so compare by real part first
// and then compare by imaginary part
static int complex_compare(const void *a, const void *b) {
  cxldouble ca = *(const cxldouble *)a;
  cxldouble cb = *(const cxldouble *)b;

  long double real_diff = cxreall(ca) - cxreall(cb);
  if (fabsl(real_diff) > 0)
    return (real_diff > 0) - (real_diff < 0);

  long double imag_diff = cximagl(ca) - cximagl(cb);
  if (fabsl(imag_diff) > 0)
    return (imag_diff > 0) - (imag_diff < 0);

  return 0;
}

static void expand_from_roots(cxldouble *coeffs,
                              const cxldouble *roots,
                              const size_t *mults,
                              size_t distinct_roots,
                              size_t coeffs_size) {
  size_t curr_deg = 0;
  for (size_t i = 0; i < coeffs_size; i++) coeffs[i] = CXL(0.0, 0.0);
  coeffs[0] = CXL(1.0, 0.0);
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
        coeffs[j] = cxaddl(cxmull(cxnegl(roots[i]), coeffs[j]), coeffs[j-1]);
      }

      // last coefficient is just multiplied by neg of the root
      // k(0) * (-c)
      coeffs[0] = cxmull(coeffs[0], cxnegl(roots[i]));
    }
  }
}

polynomial_t *polynomial_new(size_t degree) {
  polynomial_t *poly = malloc(sizeof(polynomial_t));
  if (!poly) return NULL;

  poly->degree = degree;
  poly->coeffs = calloc(degree+1, sizeof(cxldouble));
  poly->roots = calloc(degree, sizeof(cxldouble));
  poly->multiplicity = calloc(degree, sizeof(size_t));

  if (!poly->coeffs || !poly->roots || !poly->multiplicity) {
    // falied to allocate somehow
    polynomial_free(poly);
    return NULL;
  }

  // coefficients not set yet
  poly->coeffs_valid = false;

  // roots not found yet
  poly->num_distinct_roots = 0;
  poly->roots_valid = false;

  return poly;
}

polynomial_t *polynomial_from_roots(cxldouble *roots, size_t num_roots, bool dedup) {
  if (!roots || !num_roots) return NULL;
  size_t num_dist_roots = 0;
  // find the number of distinct roots
  cxldouble *distinct_roots;
  size_t *multiplicity;
  distinct_roots = calloc(num_roots, sizeof(cxldouble));
  multiplicity = calloc(num_roots, sizeof(size_t));
  if (!distinct_roots || !multiplicity) {
    if (distinct_roots) free(distinct_roots);
    if (multiplicity) free (multiplicity);
    return NULL;
  }

  if (dedup) {
    // assuming that repeated roots show up in the roots more than once
    // so we copy the roots and sort them
    cxldouble *sorted_roots;
    sorted_roots = calloc(num_roots, sizeof(cxldouble));
    if (!sorted_roots) return NULL;

    memcpy(sorted_roots, roots, num_roots * sizeof(cxldouble));
    qsort(sorted_roots, num_roots, sizeof(cxldouble), complex_compare);

    cxldouble curr_root = CXL(0.0, 0.0);
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

polynomial_t *polynomial_from_dis_roots(cxldouble *distinct_roots, size_t *mult, size_t num_dist_roots) {
  if (!distinct_roots || num_dist_roots < 1) return NULL;

  // calculate degree from the number of roots and multiplicity
  size_t degree = 0;
  for (size_t i = 0; i < num_dist_roots; i++) degree += mult[i];
  polynomial_t *poly = polynomial_new(degree);
  if (!poly) return NULL;

  memcpy(poly->roots, distinct_roots, num_dist_roots * sizeof(cxldouble));
  memcpy(poly->multiplicity, mult, num_dist_roots * sizeof(size_t));
  poly->num_distinct_roots = num_dist_roots;
  poly->roots_valid = true;

  // calculate coefficients from roots
  expand_from_roots(poly->coeffs, poly->roots, poly->multiplicity,
    poly->num_distinct_roots, poly->degree+1);
  poly->coeffs_valid = true;

  return poly;
}

polynomial_t *polynomial_from_coeffs(const cxldouble *coeffs, size_t num_coeffs) {
  if (!coeffs || num_coeffs < 1) return NULL;

  // assuming number of coefficients-1 is the degree
  polynomial_t *poly = polynomial_new(num_coeffs-1);
  if (!poly) return NULL;

  memcpy(poly->coeffs, coeffs, num_coeffs * sizeof(cxldouble));
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
  if (!poly) return NULL;

  polynomial_t *copy = polynomial_new(poly->degree);
  if (!copy) return NULL;

  memcpy(copy->coeffs, poly->coeffs, (poly->degree+1) * sizeof(cxldouble));
  memcpy(copy->roots, poly->roots, poly->num_distinct_roots * sizeof(cxldouble));
  memcpy(copy->multiplicity, poly->multiplicity, poly->num_distinct_roots * sizeof(size_t));

  copy->num_distinct_roots = poly->num_distinct_roots;
  copy->roots_valid = poly->roots_valid;
  copy->coeffs_valid = poly->coeffs_valid;
  return copy;
}

// horner's method for evaluating the polynomial
cxldouble polynomial_eval(polynomial_t *poly, cxldouble z) {
  // b_N = a_N
  // coeff N=deg+1
  cxldouble b = poly->coeffs[poly->degree];
  for (size_t i = poly->degree; i > 0; i--) {
    // b_N-1 = a_N-1 + b_N * z
    b = cxaddl(poly->coeffs[i-1], cxmull(b, z));
  }
  return b;
}

// find roots using the scaled Jenkins-Traub algorithm
bool polynomial_find_roots(polynomial_t *poly, bool dedup) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0) return false;

  cxldouble *roots = malloc(poly->degree * sizeof(cxldouble));
  if (!roots) return false;

  size_t num_roots;
  bool success = polynomial_find_roots_scaled(poly, roots, &num_roots);

  if (success && num_roots > 0) {
    // use constructor to deduplicate roots
    polynomial_t *temp = polynomial_from_roots(roots, num_roots, dedup);
    if (temp) {
      // copy deduplicated roots and multiplicities
      memcpy(poly->roots, temp->roots, temp->num_distinct_roots * sizeof(cxldouble));
      memcpy(poly->multiplicity, temp->multiplicity, temp->num_distinct_roots * sizeof(size_t));
      poly->num_distinct_roots = temp->num_distinct_roots;
      poly->roots_valid = true;
      polynomial_free(temp);
    } else {
      // deduplication failed, copy raw roots with multiplicity 1
      memcpy(poly->roots, roots, num_roots * sizeof(cxldouble));
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