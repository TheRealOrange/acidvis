//
// Created by Lin Yicheng on 1/12/25.
//

#ifndef POLYNOMIAL_UTIL_H
#define POLYNOMIAL_UTIL_H

#include <stdio.h>

float hue_to_rgb_cpu(float p, float q, float t);
void hsl_to_rgb_cpu(float h, float s, float l, unsigned char *r, unsigned char *g, unsigned char *b);

// get number of OpenMP threads (1 if OpenMP not available)
int get_num_threads(void);
void set_num_threads(int n);

size_t int_pow(size_t base, size_t exp);

size_t bsearch_le(const size_t *arr, size_t len, size_t target);
size_t bsearch_gt(const size_t *arr, size_t len, size_t target);

#endif //POLYNOMIAL_UTIL_H