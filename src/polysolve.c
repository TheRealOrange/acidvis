//
// Created by Lin Yicheng on 30/11/25.
//

#include "polynomial.h"
#include "polysolve.h"
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#define JT_EPSILON      1e-13
#define REL_EPSILON     1e-14

#define STAGE1_ITERS    5
#define STAGE2_ITERS    5
#define STAGE2_THRES    12
#define STAGE2_SCALE    19
#define MAX_STAGE3      200
#define MAX_ATTEMPTS    600

#define POLISH_ITERS    90
#define NEWTON_ITERS    80
#define CAUCHY_ITERS    80

// constants for scaling
#define BASE           2.0           // Scale factor base (power of 2 for exact FP)
#define ETA            LDBL_EPSILON  // Machine epsilon
#define INFIN          LDBL_MAX      // Largest representable number
#define SMALNO         LDBL_MIN      // Smallest normalized number

// based on CPOLY's scale() function
static long double compute_coeff_scale(polynomial_t *P) {
  size_t n = P->degree;

  // find largest and smallest moduli of coefficients
  long double hi = sqrtl(INFIN);
  long double lo = SMALNO / ETA;
  long double max_coeff = 0.0l;
  long double min_coeff = INFIN;

  for (size_t i = 0; i <= n; i++) {
    long double x = cabsl(P->coeffs[i]);
    if (x > max_coeff) max_coeff = x;
    if (x != 0.0l && x < min_coeff) min_coeff = x;
  }

  // scale only if there are very large or very small components
  if (min_coeff >= lo && max_coeff <= hi) {
    return 1.0l;
  }

  long double sc;
  long double x = lo / min_coeff;
  if (x > 1.0l) {
    sc = x;
    // check for overflow if we scale up
    if (INFIN / sc > max_coeff) sc = 1.0l;
  } else {
    // scale down using geometric mean
    sc = 1.0l / (sqrtl(max_coeff) * sqrtl(min_coeff));
  }

  // round to nearest power of BASE for exact floating point arithmetic
  int l = (int) (logl(sc) / logl(BASE) + 0.5l);
  return powl(BASE, l);
}

// compute Cauchy upper bound on root magnitudes
// such that all roots have |z| <= cauchy_bound
// bound = 1 + max(|a_i/a_n|) for i < n
long double cauchy_bound(polynomial_t *P) {
  size_t n = P->degree;
  if (n == 0) return 0.0l;

  long double an = cabsl(P->coeffs[n]);
  if (an < DBL_EPSILON) return INFINITY;

  // calculate 1 + max(|a_i/a_n|) for i < n
  long double max_ratio = 0.0;
  for (size_t i = 0; i < n; i++) {
    long double ratio = cabsl(P->coeffs[i]) / an;
    if (ratio > max_ratio) max_ratio = ratio;
  }

  return 1.0l + max_ratio;
}

// compute inner root radius, smallest B such that |a_n|B^n = sum_{i=0}^{n-1} |a_i|B^i
// estimates the magnitude of the smallest root
static long double inner_root_radius(polynomial_t *P) {
  size_t n = P->degree;
  if (n == 0) return 1.0;

  long double an = cabsl(P->coeffs[n]);
  if (an < LDBL_EPSILON) return 1.0;

  // initial guess: (|a_0|/|a_n|)^(1/n)
  long double a0 = cabsl(P->coeffs[0]);
  long double beta = (a0 > LDBL_EPSILON) ? powl(a0 / an, 1.0l / (long double)n) : 1.0l;
  if (beta < 1e-10) beta = 1e-10;
  if (beta > 1e10) beta = 1e10;

  // Newton iterations to solve |a_n|B^n = sum_{i<n} |a_i|B^i
  for (int iter = 0; iter < NEWTON_ITERS; iter++) {
    long double f = -an * powl(beta, (long double) n);
    long double df = -an * (long double)n * powl(beta, (long double) (n - 1));

    long double beta_pow = 1.0l;
    for (size_t i = 0; i < n; i++) {
      f += cabsl(P->coeffs[i]) * beta_pow;
      if (i > 0) df += cabsl(P->coeffs[i]) * (long double)i * powl(beta, (long double) (i - 1));
      beta_pow *= beta;
    }

    if (fabsl(df) < LDBL_EPSILON) break;

    long double delta = f / df;
    beta -= delta;

    if (beta < 1e-10) beta = 1e-10;
    if (fabsl(delta) < REL_EPSILON * beta) break;
  }

  return fmaxl(1e-10, beta);
}

// scale polynomial for optimal numerical stability
// returns the variable scale factor used
// and applies coefficient scaling for overflow/underflow prevention
//   P(x) -> Q(y) where y = x/s
//   Q(y) = P(s * y) = sum of { a_i (s * y)^i = Σ (a_i s^i) y^i }
// (coefficient i is multiplied by s^i)
// for each root r of Q, the root of P is s*r
long double polynomial_scale_for_roots(polynomial_t *P) {
  if (!P || P->degree == 0) return 1.0l;

  size_t n = P->degree;

  // apply coefficient scaling to prevent overflow/underflow
  long double coeff_scale = compute_coeff_scale(P);
  if (coeff_scale != 1.0l) {
    for (size_t i = 0; i <= n; i++) {
      P->coeffs[i] *= coeff_scale;
    }
  }

  // compute variable scale factor
  // use geometric mean of inner and outer Cauchy bounds
  long double inner = inner_root_radius(P);
  long double outer = cauchy_bound(P);

  // geometric mean and handle edge cases
  long double sigma;
  if (inner <= 0 || outer <= 0 || inner > outer * 1e10) {
    sigma = 1.0l;
  } else {
    sigma = sqrtl(inner * outer);
  }

  // round sigma to a power of 2 for exact arithmetic
  if (sigma != 1.0l && sigma > 0) {
    int l = (int) (logl(sigma) / logl(BASE) + 0.5l);
    sigma = powl(BASE, l);
  }

  // dont scale if sigma is close to 1 (between 0.5 and 2.0)
  // prevents numerical issues where scaling destroys significant coefficients
  if (sigma >= 0.5 && sigma <= 2.0) {
    sigma = 1.0l;
  }

  // don't scale if it would make the constant term negligible
  // relative to the leading coefficient after making monic
  if (sigma != 1.0l) {
    long double sigma_pow_n = powl(sigma, (double)n);
    long double const_mag = cabsl(P->coeffs[0]);
    long double lead_mag = cabsl(P->coeffs[n]);
    // after making monic const_new = const_orig / (lead * sigma^n)
    // we want const_new to stay significant (> LDBL_EPSILON * 1000)
    long double threshold = LDBL_EPSILON * 1000.0l;

    if (const_mag > LDBL_EPSILON * lead_mag &&
        const_mag / (lead_mag * sigma_pow_n) < threshold) {
      // compute maximum sigma that keeps constant term significant
      long double ratio = const_mag / (lead_mag * threshold);
      long double sigma_max = powl(ratio, 1.0l / (long double)n);

      if (sigma > sigma_max) {
        if (sigma_max <= 1.0l) {
          sigma = 1.0l;  // cant scale up at all
        } else {
          // round down to nearest power of 2
          int l = (int)floorl(logl(sigma_max) / logl(BASE));
          sigma = powl(BASE, l);
        }
      }
    }
  }

  // apply variable scaling: coeff[i] *= σ^i
  if (sigma != 1.0l) {
    long double sigma_pow = 1.0l;
    for (size_t i = 0; i <= n; i++) {
      P->coeffs[i] *= sigma_pow;
      sigma_pow *= sigma;
    }
  }

  return sigma;
}

// unscale a root found from scaled polynomial
// if we found root r from scaled polynomial, actual root is sigma * r
static complex long double unscale_root(const complex long double scaled_root, const long double sigma) {
  return scaled_root * sigma;
}

polynomial_t *H_0(polynomial_t *P) {
  if (!P || P->degree == 0) return nullptr;
  polynomial_t *H = polynomial_new(P->degree - 1);

  // iterate N = degree times to take derivative
  for (size_t i = P->degree; i > 0; i--) {
    // H(N-1) = (k(N) * N) * x ^ (N-1)
    H->coeffs[i - 1] = P->coeffs[i] * (long double)i;
  }

  H->coeffs_valid = true;
  return H;
}

// calculate the normalized next H polynomial AND next shift value
jt_status H_norm_next(polynomial_t *H, polynomial_t *P, complex long double *k_p, complex long double *k_h,
                      complex long double shift, complex long double *next_shift, polynomial_t **H_out) {
  if (!P || !H || P->degree == 0 || H->degree == 0 || !H_out)
    return JT_ERROR;

  // intermediate b_N polynomials for horner's method
  // forming P(x) = (b_N + b_N-1 * x^1 + b_N-2 * x^2 + ... b_0 * x ^ (N-1))(x - shift) + P(x_0)
  // degree of the p intermediate is p_deg-1 so num coeffs is p_deg
  // degree of the h intermediate is h_deg-1 so num coeffs is h_deg
  size_t p_deg = P->degree;
  size_t h_deg = H->degree;

  // use horner's method to evaluate the polynomials
  // and simultaneously get the intermediate coefficients
  complex long double b_p = P->coeffs[p_deg];
  for (size_t i = p_deg; i > 0; i--) {
    k_p[i - 1] = b_p;
    b_p = P->coeffs[i - 1] + b_p * shift;
  } // b_p = P(shift)

  complex long double b_h = H->coeffs[h_deg];
  for (size_t i = h_deg; i > 0; i--) {
    k_h[i - 1] = b_h;
    b_h = H->coeffs[i - 1] + b_h * shift;
  } // b_h = H(shift)

  // check if found a root
  if (cabsl(b_p) < JT_EPSILON * (long double)p_deg) {
    *H_out = nullptr;
    if (next_shift) *next_shift = shift;
    return JT_CONVERGED;
  }

  // else, proceed as normal and compute normalized H_next
  // this is actually norm * H(s) (multiply H(s) instead in the loop)
  complex long double norm = -b_h * k_p[p_deg - 1];
  if (cabsl(norm) < JT_EPSILON * cabsl(k_p[p_deg - 1])) {
    *H_out = nullptr;
    return JT_H_ZERO;
  }

  // degree of output should be p_deg-1 so p_deg coefficients
  // reuse k_p for output
  // get normalized coefficients by dividing out leading
  k_p[p_deg-1] = 1.0;
  for (size_t i = 0; i < p_deg-1; i++) { // h is one degree smaller
    k_p[i] = (k_h[i] * b_p - b_h * k_p[i]) / norm;
  }

  for (size_t i = 0; i < p_deg; i++) {
    if (!isfinite(creall(k_p[i])) || !isfinite(cimagl(k_p[i]))) {
      return JT_ERROR;  // overflow detected, try different starting point
    }
  }

  // construct new output H polynomial as the max degree required
  polynomial_t *hnext = polynomial_from_coeffs(k_p, p_deg);

  if (!hnext) return JT_ERROR;

  if (next_shift) {
    complex long double h_at_s = polynomial_eval(hnext, shift);
    if (cabsl(h_at_s) > JT_EPSILON * (long double)(p_deg - 1)) {
        *next_shift = shift - b_p / h_at_s;
    } else {
        *next_shift = shift;
    }
  }

  *H_out = hnext;
  return JT_OK;
}

// find next root of a scaled polynomial, roots should be near unit magnitude
jt_status find_next_root(polynomial_t *P, complex long double *root_out) {
  if (!P || !root_out || P->degree == 0) return JT_ERROR;

  complex long double *work_p = calloc(P->degree, sizeof(complex long double));
  complex long double *work_h = calloc(P->degree, sizeof(complex long double));
  if (!work_p || !work_h) {
    if (work_p) free(work_p);
    if (work_h) free(work_h);
    return JT_ERROR;
  }

  // degree 1 base case
  if (P->degree == 1) {
    *root_out = -P->coeffs[0] / P->coeffs[1];
    free(work_p);
    free(work_h);
    return JT_OK;
  }

  // degree 2 use quadratic formula
  if (P->degree == 2) {
    complex long double a = P->coeffs[2];
    complex long double b = P->coeffs[1];
    complex long double c = P->coeffs[0];
    complex long double disc = csqrtl(b * b - 4.0l * a * c);

    // choose the sign that avoids cancellation
    complex long double q;
    if (cabsl(b + disc) > cabsl(b - disc)) {
      q = -0.5l * (b + disc);
    } else {
      q = -0.5l * (b - disc);
    }

    complex long double r1 = q / a;
    complex long double r2 = c / q;

    // return smaller magnitude root first (Jenkins-Traub convention)
    *root_out = (cabsl(r1) < cabsl(r2)) ? r1 : r2;

    free(work_p);
    free(work_h);
    return JT_OK;
  }

  // for scaled polynomial, roots are near unit magnitude
  // Start search on unit circle
  long double beta = inner_root_radius(P);

  // try different starting angles
  for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    // shift on circle of inner root radius
    // CPOLY uses 94 degrees = 94 * pi / 180 = 1.6406 radians
    long double angle = 1.6406094968746698l + (long double)attempt * 0.1l;
    complex long double s = beta * cexp(I * angle);

    // initialize H_0 = P'
    polynomial_t *H = H_0(P);
    if (!H) {
      free(work_p);
      free(work_h);
      return JT_ERROR;
    }

    // stage 1 no shift steps, shift = 0
    for (int i = 0; i < STAGE1_ITERS; i++) {
      polynomial_t *H_new;
      jt_status st = H_norm_next(H, P, work_p, work_h, 0, nullptr, &H_new);
      polynomial_free(H);
      if (st == JT_ERROR) { H = nullptr; break; }
      if (st == JT_H_ZERO) { H = H_0(P); continue; }  // reset
      H = H_new;
    }
    if (!H) continue;

    // stage 2 fixed shift
    int h_zero_count = 0;
    size_t max_iters = ((P->degree > STAGE2_THRES) ? P->degree / STAGE2_SCALE : 0) + STAGE2_ITERS;
    for (size_t i = 0; i < max_iters; i++) {
      polynomial_t *H_new;
      jt_status st = H_norm_next(H, P, work_p, work_h, s, nullptr, &H_new);

      // skip to next step (is this correct?)
      if (st == JT_CONVERGED) {
        *root_out = s;
        polynomial_free(H);
        free(work_p);
        free(work_h);
        return JT_OK;
      }

      if (st == JT_H_ZERO) {
        // perturb and retry
        s *= 1.001;
        s += 0.001 * cexp(I * h_zero_count);
        h_zero_count++;
        if (h_zero_count > 5) break;  // give up on this starting point
        continue;
      }

      if (st == JT_ERROR) {
        H = nullptr; break;
      }

      if (H_new) {
        polynomial_free(H);
        H = H_new;
      }
    }
    if (!H) continue;

    h_zero_count = 0;
    complex long double s_new = s;
    int stagnant_count = 0;
    long double prev_delta = INFINITY;
    for (int i = 0; i < MAX_STAGE3; i++) {
      polynomial_t *H_new;
      jt_status st = H_norm_next(H, P, work_p, work_h, s, &s_new, &H_new);

      if (st == JT_CONVERGED) {
        *root_out = s;
        polynomial_free(H);
        free(work_p);
        free(work_h);
        return JT_OK;
      }

      if (st == JT_H_ZERO) {
        // perturb and retry
        s *= 1.001;
        s += 0.001 * cexp(I * h_zero_count);
        h_zero_count++;
        if (h_zero_count > 5) break;  // give up on this starting point
        continue;
      }

      if (st == JT_ERROR) break;
      if (H_new) {
        polynomial_free(H);
        H = H_new;
      }

      // check convergence
      long double delta = cabsl(s_new - s);
      if (delta < REL_EPSILON * fmaxl(1.0, cabsl(s_new))) {
        // verify it's actually a root
        complex long double p_val = polynomial_eval(P, s_new);
        long double p_scale = 0;
        for (size_t j = 0; j <= P->degree; j++) {
          long double m = cabsl(P->coeffs[j]);
          if (m > p_scale) p_scale = m;
        }
        long double p_tol = JT_EPSILON * p_scale * (long double)P->degree;
        if (cabsl(p_val) < p_tol) {
          *root_out = s_new;
          polynomial_free(H);
          free(work_p);
          free(work_h);
          return JT_OK;
        }
      }

      if (delta >= prev_delta) {
        stagnant_count++;
      } else {
        stagnant_count = 0;
      }

      if (stagnant_count > 20) break;  // Not converging, try new starting point

      prev_delta = delta;
      s = s_new;
    }

    polynomial_free(H);
  }

  free(work_p);
  free(work_h);
  return JT_ERROR; // failed to find next root
}

// use horner's method to deflate polynomial
polynomial_t *polynomial_deflate(polynomial_t *P, complex long double root) {
  if (!P || P->degree == 0) return nullptr;

  size_t p_deg = P->degree;
  // degree of the p deflated is p_deg-1 so num coeffs is p_deg
  complex long double *a = P->coeffs;
  complex long double *b = calloc(p_deg, sizeof(complex long double));
  if (!b) {
    return nullptr;
  }

  // use horner's method to evaluate the polynomials
  // and simultaneously get the intermediate coefficients
  // b_{n-1} = a_n
  b[p_deg-1] = a[p_deg];

  // b_k = a_{k+1} + root * b_{k+1} for k = n-2 down to 0
  for (size_t k = p_deg-1; k > 0; k--) {
    b[k - 1] = a[k] + root * b[k];
  } // b_n = P(root)

  polynomial_t *deflated = polynomial_from_coeffs(b, p_deg);
  free(b);
  return deflated;
}

// polish a root using Newton's method on the original unscaled polynomial
static complex long double polish_root(polynomial_t *P, complex long double root) {
  complex long double z = root;

  for (int iter = 0; iter < POLISH_ITERS; iter++) {
    // evaluate P(z) and P'(z)
    complex long double p_val = P->coeffs[P->degree];
    complex long double dp_val = 0;
    for (size_t i = P->degree; i > 0; i--) {
      dp_val = dp_val * z + p_val;
      p_val = P->coeffs[i-1] + p_val * z;
    }

    if (cabsl(dp_val) < LDBL_EPSILON) break;

    complex long double delta = p_val / dp_val;
    z = z - delta;

    if (cabsl(delta) < REL_EPSILON * fmaxl(1.0l, cabsl(z))) break;
  }

  return z;
}

bool polynomial_find_roots_scaled(polynomial_t *poly, complex long double *roots, size_t *num_roots) {
  if (!poly || !poly->coeffs_valid || poly->degree == 0 || !roots || !num_roots)
    return false;

  // keep a copy of the original for root polishing
  polynomial_t *original = polynomial_copy(poly);
  if (!original) return false;

  // work with a copy for deflation
  polynomial_t *work = polynomial_copy(poly);
  if (!work) {
    polynomial_free(original);
    return false;
  }

  size_t roots_found = 0;

  // Remove zeros at the origin before scaling
  // avoids numerical issues where scaling reduces constants to near-zero
  // zero at origin means constant term is zero relative to polynomial scale
  long double orig_scale = 0;
  for (size_t i = 0; i <= work->degree; i++) {
    long double m = cabsl(work->coeffs[i]);
    if (m > orig_scale) orig_scale = m;
  }
  long double zero_tol = orig_scale * LDBL_EPSILON * 10.0l;  // conservative tolerance

  while (work->degree > 0 && cabsl(work->coeffs[0]) < zero_tol) {
    roots[roots_found++] = 0.0;
    // shift coefficients down (divide by x)
    for (size_t i = 0; i < work->degree; i++) {
      work->coeffs[i] = work->coeffs[i + 1];
    }
    work->degree--;
  }

  // scale the polynomial (store the scale factor)
  long double sigma = polynomial_scale_for_roots(work);

  // make monic
  complex long double lead = work->coeffs[work->degree];
  for (size_t i = 0; i <= work->degree; i++) {
    work->coeffs[i] /= lead;
  }

  while (work->degree > 0) {
    complex long double scaled_root;

    jt_status st = find_next_root(work, &scaled_root);
    if (st != JT_OK) {
      polynomial_free(work);
      polynomial_free(original);
      *num_roots = roots_found;
      return roots_found > 0; // partial success
    }

    // remember to unscale
    complex long double raw_root = unscale_root(scaled_root, sigma);

    // polish the root on the polynomial
    complex long double polished_root = polish_root(original, raw_root);
    complex long double polished_scaled_root = polish_root(work, scaled_root);
    roots[roots_found++] = polished_root;

    // deflate again using the scaled root (more accurate for the scaled polynomial)
    polynomial_t *deflated = polynomial_deflate(work, polished_scaled_root);
    polynomial_free(work);
    work = deflated;

    if (!work && roots_found < poly->degree) {
      polynomial_free(original);
      *num_roots = roots_found;
      return roots_found > 0; // partial success
    }
  }

  polynomial_free(work);
  polynomial_free(original);
  *num_roots = roots_found;
  return true;
}







