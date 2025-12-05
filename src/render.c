//
// Created by Lin Yicheng on 30/11/25.
// Refactored: common rendering interface
//

#include "render.h"
#include "render_internal.h"

#include <SDL3/SDL.h>
#include <stdlib.h>

// shared render state declared extern in render_internal.h
float gamma_value = 0.1f;
bool invert_colors = false;
bool use_log_scale = false;
float pix_scale = 0.0f;
float x_offset = 0.0f;
float y_offset = 0.0f;
polynomial_t *current_polynomial = NULL;

size_t frame_width = 0;
size_t frame_height = 0;
size_t total_pixels = 0;
float global_max_mag_host = 0.0f;

unsigned char *host_pixels = NULL;

// cpu hue accumulation buffers
float *hue_x_val = NULL;
float *hue_y_val = NULL;
int *hue_count = NULL;

// gpu initialization flag
bool gpu_initialized = false;

bool initialize_renderer(void) {
#ifdef HAVE_OPENCL
  SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "attempting opencl initialization");
  if (render_gpu_init()) {
    return true;
  }
  SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "opencl initialization failed, using cpu fallback");
#else
  SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "opencl not available, using cpu renderer");
#endif
  return true;
}

void cleanup_renderer(void) {
  SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "cleaning up renderer");

#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    render_gpu_cleanup();
  }
#endif

  if (host_pixels) {
    free(host_pixels);
    host_pixels = NULL;
  }

  if (hue_x_val) {
    free(hue_x_val);
    hue_x_val = NULL;
  }
  if (hue_y_val) {
    free(hue_y_val);
    hue_y_val = NULL;
  }
  if (hue_count) {
    free(hue_count);
    hue_count = NULL;
  }
}

void set_gamma(float gamma) {
  gamma_value = gamma;
}

float get_gamma(void) {
  return gamma_value;
}

void toggle_log_scale(void) {
  use_log_scale = !use_log_scale;
}

bool is_log_scale(void) {
  return use_log_scale;
}

void toggle_invert(void) {
  invert_colors = !invert_colors;
}

bool get_invert(void) {
  return invert_colors;
}

void set_polynomial(polynomial_t *P) {
  current_polynomial = P;

#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    render_gpu_set_polynomial(P);
  }
#endif
}

void resize_buffers(int width, int height) {
  if (frame_width == (size_t)width && frame_height == (size_t)height) {
    return;
  }

  SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "resizing buffers: %dx%d", width, height);

  frame_width = width;
  frame_height = height;
  total_pixels = (size_t)width * (size_t)height;

  // reallocate host pixel buffer
  if (host_pixels) free(host_pixels);
  host_pixels = malloc(total_pixels * 4);

  // cpu fallback buffers
  if (!gpu_initialized) {
    if (hue_x_val) free(hue_x_val);
    if (hue_y_val) free(hue_y_val);
    if (hue_count) free(hue_count);

    hue_x_val = malloc(total_pixels * sizeof(float));
    hue_y_val = malloc(total_pixels * sizeof(float));
    hue_count = malloc(total_pixels * sizeof(int));
  }

#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    render_gpu_resize_buffers(width, height);
  }
#endif
}

void render_frame_roots(float scale_inp, float x_off_inp, float y_off_inp) {
  if (!current_polynomial) return;

  pix_scale = scale_inp;
  x_offset = x_off_inp;
  y_offset = y_off_inp;

#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    render_frame_gpu();
  } else {
    render_frame_cpu();
  }
#else
  render_frame_cpu();
#endif
}

void render_point_cloud(cxldouble *roots, bool *valid,
                        size_t num_perms, size_t stride, float point_radius,
                        float scale, float x_off, float y_off) {
#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    render_point_cloud_gpu(roots, valid, num_perms, stride, point_radius,
                           scale, x_off, y_off);
  } else {
    render_point_cloud_cpu(roots, valid, num_perms, stride, point_radius,
                           scale, x_off, y_off);
  }
#else
  render_point_cloud_cpu(roots, valid, num_perms, stride, point_radius,
                         scale, x_off, y_off);
#endif
}

void clear_frame_buffer(unsigned char r, unsigned char g, unsigned char b) {
  if (!host_pixels) return;

  for (size_t i = 0; i < total_pixels; i++) {
    host_pixels[i * 4 + 0] = r;
    host_pixels[i * 4 + 1] = g;
    host_pixels[i * 4 + 2] = b;
    host_pixels[i * 4 + 3] = 255;
  }
}

const unsigned char *get_pixel_data(void) {
  return host_pixels;
}

float get_max_magnitude(void) {
  return global_max_mag_host;
}

const char *get_device_name(void) {
#ifdef HAVE_OPENCL
  if (gpu_initialized) {
    extern const char *render_gpu_get_device_name(void);
    return render_gpu_get_device_name();
  }
#endif
  return "cpu (fallback)";
}
