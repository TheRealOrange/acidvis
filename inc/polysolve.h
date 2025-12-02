//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_POLYSOLVE_H
#define POLYNOMIAL_POLYSOLVE_H

#include "polynomial.h"

typedef enum {
  JT_OK,
  JT_CONVERGED,      // P(s) approx 0 (found root)
  JT_H_ZERO,         // H(s) approx 0
  JT_ERROR
} jt_status;

polynomial_t *H_0(polynomial_t *P);
jt_status H_norm_next(polynomial_t *H, polynomial_t *P, complex long double *k_p, complex long double *k_h,
                      complex long double shift, complex long double *next_shift, polynomial_t **H_out);

long double cauchy_bound(polynomial_t *P);

jt_status find_next_root(polynomial_t *P, complex long double *root_out);
bool polynomial_find_roots_scaled(polynomial_t *poly, complex long double *roots, size_t *num_roots);
polynomial_t *polynomial_deflate(polynomial_t *P, complex long double root);

#endif //POLYNOMIAL_POLYSOLVE_H