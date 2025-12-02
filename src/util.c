//
// Created by Lin Yicheng on 1/12/25.
//

#include "util.h"

#include <stdint.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// cpu hsl helpers

float hue_to_rgb_cpu(float p, float q, float t) {
  if (t < 0.0f) t += 1.0f;
  if (t > 1.0f) t -= 1.0f;
  if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
  if (t < 0.5f) return q;
  if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  return p;
}

void hsl_to_rgb_cpu(float h, float s, float l, unsigned char *r, unsigned char *g, unsigned char *b) {
  float rf, gf, bf;

  if (s < 0.001f) {
    rf = gf = bf = l;
  } else {
    float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    float p = 2.0f * l - q;
    rf = hue_to_rgb_cpu(p, q, h + 1.0f / 3.0f);
    gf = hue_to_rgb_cpu(p, q, h);
    bf = hue_to_rgb_cpu(p, q, h - 1.0f / 3.0f);
  }

  *r = (unsigned char)(rf * 255.0f);
  *g = (unsigned char)(gf * 255.0f);
  *b = (unsigned char)(bf * 255.0f);
}

int get_num_threads(void) {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void set_num_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}

size_t int_pow(size_t base, size_t exp) {
  size_t result = 1;
  for (size_t i = 0; i < exp; i++) {
    result *= base;
  }
  return result;
}

static size_t bsearch_helper(const size_t *arr, size_t len, size_t target) {
  size_t lo = 0;
  size_t hi = len;

  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (arr[mid] <= target)
      lo = mid + 1;
    else
      hi = mid;
  }

  return lo;
}

// find largest value in arr that is <= target
size_t bsearch_le(const size_t *arr, size_t len, size_t target) {
  if (len == 0) return SIZE_MAX;
  size_t lo = bsearch_helper(arr, len, target);
  // lo is now the first index where arr[lo] > target
  // so lo - 1 is the last index where arr[lo-1] <= target
  return (lo == 0) ? SIZE_MAX : lo - 1;
}

// find smallest value in arr that is > target
size_t bsearch_gt(const size_t *arr, size_t len, size_t target) {
  if (len == 0) return SIZE_MAX;
  size_t lo = bsearch_helper(arr, len, target);
  // lo is now the first index where arr[lo] > target
  return (lo == len) ? SIZE_MAX : lo;
}