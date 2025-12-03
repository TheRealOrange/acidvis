//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_RENDER_INTERNAL_H
#define POLYNOMIAL_RENDER_INTERNAL_H

#include "render.h"
#include "polynomial.h"
#include "compat_complex.h"

// shared render state
extern float gamma_value;
extern bool invert_colors;
extern bool use_log_scale;
extern float pix_scale;
extern float x_offset, y_offset;
extern polynomial_t *current_polynomial;

extern size_t frame_width;
extern size_t frame_height;
extern size_t total_pixels;
extern float global_max_mag_host;

extern unsigned char *host_pixels;

// cpu hue accumulation buffers
extern float *hue_x_val;
extern float *hue_y_val;
extern int *hue_count;

// gpu initialization state
extern bool gpu_initialized;

// cpu rendering functions
void render_frame_cpu(void);
void render_point_cloud_cpu(cxldouble *roots, const size_t *num_distinct,
                            size_t num_perms, size_t stride, float point_radius,
                            float scale, float x_off, float y_off);

#ifdef HAVE_OPENCL

// gpu rendering functions
bool render_gpu_init(void);
void render_gpu_cleanup(void);
void render_gpu_resize_buffers(int width, int height);
void render_gpu_set_polynomial(polynomial_t *P);

void render_frame_gpu(void);
void render_point_cloud_gpu(cxldouble *roots, const size_t *num_distinct,
                            size_t num_perms, size_t stride, float point_radius,
                            float scale, float x_off, float y_off);

#endif // HAVE_OPENCL

#endif // POLYNOMIAL_RENDER_INTERNAL_H
