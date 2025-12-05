//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_RENDER_H
#define POLYNOMIAL_RENDER_H

#include <stdbool.h>
#include <stddef.h>
#include "compat_complex.h"

#ifdef HAVE_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#endif

struct polynomial;
typedef struct polynomial polynomial_t;

bool initialize_renderer(void);
void cleanup_renderer(void);
void set_polynomial(polynomial_t *P);

// get/set the gamma to render
void set_gamma(float gamma);
float get_gamma(void);

// toggle/get color inversion
void toggle_invert(void);
bool get_invert(void);

// toggle/get log scale eenabled
void toggle_log_scale(void);
bool is_log_scale(void);

// resize internal buffers when window size changes
void resize_buffers(int width, int height);

// render a frame with the given view parameters
// scale: pixel scale, smaller -> zoomed out, larger -> zoomed in
// x_off, y_off: offset in the complex plane
void render_frame_roots(float scale, float x_off, float y_off);
void clear_frame_buffer(unsigned char r, unsigned char g, unsigned char b);

// render the roots point cloud given view parameters
void render_point_cloud(cxldouble *roots, bool *valid, size_t num_perms, size_t stride, float point_radius,
                        float scale, float x_off, float y_off);

// get the pixel data after rendering (rgba format, 4 bytes per pixel)
const unsigned char* get_pixel_data(void);

// get the maximum magnitude from the last render
float get_max_magnitude(void);

// get the name of the rendering device
const char* get_device_name(void);

// frame dimensions set by resize_buffers
extern size_t frame_width;
extern size_t frame_height;
extern size_t total_pixels;

#endif //POLYNOMIAL_RENDER_H