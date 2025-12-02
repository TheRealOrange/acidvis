//
// Created by Lin Yicheng on 30/11/25.
//

#include "render.h"
#include "polynomial.h"
#include "generated/kernel_source.h"

#include <stdio.h>
#ifdef HAVE_OPENCL
#include "generated/kernel_source.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <float.h>

#include "util.h"

static float gamma_value = 0.1f;
static bool invert_colors = false;
static bool use_log_scale = false;
static float pix_scale;
static float x_offset, y_offset;
static polynomial_t *current_polynomial = nullptr;

size_t frame_width = 0;
size_t frame_height = 0;
size_t total_pixels = 0;
float global_max_mag_host = 0.0f;

float *hue_x_val= nullptr;
float *hue_y_val = nullptr;
int *hue_count = nullptr;

static unsigned char *host_pixels = nullptr;

#ifdef HAVE_OPENCL

bool initialized = false;
cl_context context = nullptr;
cl_command_queue queue = nullptr;
cl_program program = nullptr;
cl_device_id device = nullptr;
char device_name[256];

cl_mem d_poly_coeff_real = nullptr;
cl_mem d_poly_coeff_imag = nullptr;

cl_mem d_buffer_real = nullptr;
cl_mem d_buffer_imag = nullptr;
cl_mem d_buffer_output_frame = nullptr;

cl_kernel kernel_process = nullptr;
cl_kernel kernel_magnitude_reduction = nullptr;
cl_kernel kernel_reduce_final = nullptr;
cl_kernel kernel_render_frame = nullptr;
cl_kernel kernel_point_cloud = nullptr;
cl_kernel kernel_clear_hue = nullptr;
cl_kernel kernel_point_cloud_hue = nullptr;

cl_mem d_buffer_partial_max = nullptr;
cl_mem d_buffer_final_max = nullptr;

cl_mem d_perm_roots_real = nullptr;
cl_mem d_perm_roots_imag = nullptr;
cl_mem d_perm_distinct = nullptr;
cl_mem d_perm_roots_hue_ind = nullptr;
cl_mem d_perm_roots_count = nullptr;
size_t d_perms_capacity = 0;

size_t reduction_workgroup_size = 256;
size_t num_reduction_workgroups = 0;

#endif

bool initialize_renderer(void) {
#ifdef HAVE_OPENCL
  cl_int err;
  cl_platform_id platform = nullptr;

  err = clGetPlatformIDs(1, &platform, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to get opencl platform\n");
    return false;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "failed to get opencl device\n");
      return false;
    }
  }

  err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
  if (err != CL_SUCCESS) {
    strcpy(device_name, "unknown device");
  }
  printf("using opencl device: %s\n", device_name);

  context = clCreateContext(nullptr, 1, &device, nullptr, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create opencl context\n");
    return false;
  }

#ifdef CL_VERSION_2_0
  cl_queue_properties properties[] = {0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
#else
  queue = clCreateCommandQueue(context, device, 0, &err);
#endif
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create command queue\n");
    return false;
  }

  const char *src_ptr = kernel_source;
  size_t src_len = strlen(kernel_source);
  program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create program\n");
    return false;
  }

  err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, NULL);
  if (err != CL_SUCCESS) {
    char build_log[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(build_log), build_log, nullptr);
    fprintf(stderr, "opencl build error:\n%s\n", build_log);
    return false;
  }

  kernel_process = clCreateKernel(program, "evaluate_polynomial", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create evaluate_polynomial kernel\n");
    return false;
  }

  kernel_magnitude_reduction = clCreateKernel(program, "compute_max_magnitude", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create compute_max_magnitude kernel\n");
    return false;
  }

  kernel_reduce_final = clCreateKernel(program, "reduce_max_final", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create reduce_max_final kernel\n");
    return false;
  }

  kernel_render_frame = clCreateKernel(program, "render_frame", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create render_frame kernel\n");
    return false;
  }

  kernel_point_cloud = clCreateKernel(program, "render_point_cloud", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create render_point_cloud kernel\n");
    return false;
  }

  kernel_clear_hue = clCreateKernel(program, "clear_cloud_hue", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create clear_cloud_hue kernel\n");
    return false;
  }

  kernel_point_cloud_hue = clCreateKernel(program, "render_point_cloud_hue", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create render_point_cloud_hue kernel\n");
    return false;
  }

  d_buffer_final_max = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create final max buffer\n");
    return false;
  }

  initialized = true;
#endif

  return true;
}

void cleanup_renderer(void) {
#ifdef HAVE_OPENCL
  if (kernel_process) clReleaseKernel(kernel_process);
  if (kernel_magnitude_reduction) clReleaseKernel(kernel_magnitude_reduction);
  if (kernel_reduce_final) clReleaseKernel(kernel_reduce_final);
  if (kernel_render_frame) clReleaseKernel(kernel_render_frame);
  if (kernel_point_cloud) clReleaseKernel(kernel_point_cloud);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  if (d_buffer_real) clReleaseMemObject(d_buffer_real);
  if (d_buffer_imag) clReleaseMemObject(d_buffer_imag);
  if (d_buffer_output_frame) clReleaseMemObject(d_buffer_output_frame);
  if (d_poly_coeff_real) clReleaseMemObject(d_poly_coeff_real);
  if (d_poly_coeff_imag) clReleaseMemObject(d_poly_coeff_imag);
  if (d_buffer_partial_max) clReleaseMemObject(d_buffer_partial_max);
  if (d_buffer_final_max) clReleaseMemObject(d_buffer_final_max);
  if (d_perm_roots_real) clReleaseMemObject(d_perm_roots_real);
  if (d_perm_roots_imag) clReleaseMemObject(d_perm_roots_imag);
#endif

  if (host_pixels) {
    free(host_pixels);
    host_pixels = nullptr;
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
  if (!initialized || !P) return;

  cl_int err;
  size_t num_coeffs = P->degree + 1;

  if (d_poly_coeff_real) clReleaseMemObject(d_poly_coeff_real);
  if (d_poly_coeff_imag) clReleaseMemObject(d_poly_coeff_imag);

  d_poly_coeff_real = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      num_coeffs * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create coefficient real buffer\n");
    return;
  }

  d_poly_coeff_imag = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      num_coeffs * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create coefficient imag buffer\n");
    return;
  }

  float *coeffs_real = malloc(num_coeffs * sizeof(float));
  float *coeffs_imag = malloc(num_coeffs * sizeof(float));

  for (size_t i = 0; i < num_coeffs; i++) {
    coeffs_real[i] = (float)creall(P->coeffs[i]);
    coeffs_imag[i] = (float)cimagl(P->coeffs[i]);
  }

  err = clEnqueueWriteBuffer(queue, d_poly_coeff_real, CL_FALSE, 0,
                             num_coeffs * sizeof(float), coeffs_real, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to write coefficient real buffer\n");
  }

  err = clEnqueueWriteBuffer(queue, d_poly_coeff_imag, CL_FALSE, 0,
                             num_coeffs * sizeof(float), coeffs_imag, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to write coefficient imag buffer\n");
  }

  clFinish(queue);

  free(coeffs_real);
  free(coeffs_imag);
#endif
}

void resize_buffers(int width, int height) {
  if (frame_width == (size_t)width && frame_height == (size_t)height) {
    return;
  }

  frame_width = width;
  frame_height = height;
  total_pixels = width * height;

  if (host_pixels) free(host_pixels);
  host_pixels = malloc(total_pixels * 4);

  if (!initialized) {
    // cpu fallback
    if (hue_x_val) free(hue_x_val);
    if (hue_y_val) free(hue_y_val);
    if (hue_count) free(hue_count);

    hue_x_val = malloc(total_pixels * sizeof(float));
    hue_y_val = malloc(total_pixels * sizeof(float));
    hue_count = malloc(total_pixels * sizeof(int));
  }

#ifdef HAVE_OPENCL
  if (!initialized) return;

  cl_int err;

  if (d_buffer_real) clReleaseMemObject(d_buffer_real);
  if (d_buffer_imag) clReleaseMemObject(d_buffer_imag);
  if (d_buffer_output_frame) clReleaseMemObject(d_buffer_output_frame);
  if (d_buffer_partial_max) clReleaseMemObject(d_buffer_partial_max);
  if (d_perm_roots_hue_ind) clReleaseMemObject(d_perm_roots_hue_ind);
  if (d_perm_roots_count) clReleaseMemObject(d_perm_roots_count);

  d_buffer_real = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  total_pixels * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create real buffer\n");
    return;
  }

  d_buffer_imag = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  total_pixels * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create imag buffer\n");
    return;
  }

  d_buffer_output_frame = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         total_pixels * 4 * sizeof(unsigned char), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create output frame buffer\n");
    return;
  }

  num_reduction_workgroups = (total_pixels + reduction_workgroup_size - 1) / reduction_workgroup_size;

  d_buffer_partial_max = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        num_reduction_workgroups * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create partial max buffer\n");
    return;
  }

  d_perm_roots_hue_ind = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        total_pixels * 2 * sizeof(cl_float), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create perm num distinct buffer\n");
    return;
  }

  d_perm_roots_count = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        total_pixels * sizeof(cl_int), NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to create perm num distinct buffer\n");
    return;
  }
#endif
}

static void render_frame_cpu(void) {
  if (!current_polynomial || !host_pixels) return;

  float max_mag = 0.0f;

  float *mags = malloc(total_pixels * sizeof(float));
  float *args = malloc(total_pixels * sizeof(float));

  for (size_t y = 0; y < frame_height; y++) {
    for (size_t x = 0; x < frame_width; x++) {
      float real_x = ((float)x - (float)frame_width * 0.5f) * pix_scale + x_offset;
      float real_y = ((float)y - (float)frame_height * 0.5f) * pix_scale + y_offset;
      complex long double z = real_x + real_y * I;

      complex long double result = polynomial_eval(current_polynomial, z);

      size_t idx = y * frame_width + x;
      float mag = (float)cabsl(result);
      float arg = (float)cargl(result);
      if (!isfinite(mag)) mag = FLT_MAX;

      mags[idx] = mag;
      args[idx] = arg;

      if (mag > max_mag) max_mag = mag;
    }
  }

  global_max_mag_host = max_mag;

  float log_max = log1pf(max_mag);

  for (size_t i = 0; i < total_pixels; i++) {
    float hue = (args[i] + (float)M_PI) / (2.0f * (float)M_PI);

    float lightness;
    float sat;
    if (max_mag < 1e-10f) {
      lightness = 1.0f;
      sat = 0.0f;
    } else {
      float normalized;
      if (use_log_scale) {
        float log_mag = log1pf(mags[i]);
        normalized = log_mag / log_max;
      } else {
        normalized = mags[i] / max_mag;
      }
      if (normalized > 1.0f) normalized = 1.0f;
      if (normalized < 0.0f) normalized = 0.0f;

      float high_contrast_normalized = powf(normalized, gamma_value);

      lightness = 1.0f - high_contrast_normalized * 0.93f;
      if (invert_colors) {
        lightness = 1.0f - lightness;
      }

      sat = 0.1f + high_contrast_normalized * 0.7f;
    }

    unsigned char r, g, b;
    hsl_to_rgb_cpu(hue, sat, lightness, &r, &g, &b);

    host_pixels[i * 4 + 0] = r;
    host_pixels[i * 4 + 1] = g;
    host_pixels[i * 4 + 2] = b;
    host_pixels[i * 4 + 3] = 255;
  }

  free(mags);
  free(args);
}

#ifdef HAVE_OPENCL
static void render_frame_gpu(void) {
  if (!current_polynomial || !initialized) return;

  cl_int err;

  size_t global_work_size_2d[2] = {frame_width, frame_height};
  cl_int num_coeffs = (cl_int)(current_polynomial->degree + 1);
  cl_int fw = (cl_int)frame_width;
  cl_int fh = (cl_int)frame_height;

  clSetKernelArg(kernel_process, 0, sizeof(cl_mem), &d_buffer_real);
  clSetKernelArg(kernel_process, 1, sizeof(cl_mem), &d_buffer_imag);
  clSetKernelArg(kernel_process, 2, sizeof(cl_mem), &d_poly_coeff_real);
  clSetKernelArg(kernel_process, 3, sizeof(cl_mem), &d_poly_coeff_imag);
  clSetKernelArg(kernel_process, 4, sizeof(cl_int), &num_coeffs);
  clSetKernelArg(kernel_process, 5, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_process, 6, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_process, 7, sizeof(float), &pix_scale);
  clSetKernelArg(kernel_process, 8, sizeof(float), &x_offset);
  clSetKernelArg(kernel_process, 9, sizeof(float), &y_offset);

  err = clEnqueueNDRangeKernel(queue, kernel_process, 2, nullptr,
                                global_work_size_2d, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue evaluate_polynomial: %d\n", err);
    return;
  }

  size_t global_size_reduction = num_reduction_workgroups * reduction_workgroup_size;
  int tp = (int)total_pixels;

  clSetKernelArg(kernel_magnitude_reduction, 0, sizeof(cl_mem), &d_buffer_real);
  clSetKernelArg(kernel_magnitude_reduction, 1, sizeof(cl_mem), &d_buffer_imag);
  clSetKernelArg(kernel_magnitude_reduction, 2, reduction_workgroup_size * sizeof(float), nullptr);
  clSetKernelArg(kernel_magnitude_reduction, 3, sizeof(cl_mem), &d_buffer_partial_max);
  clSetKernelArg(kernel_magnitude_reduction, 4, sizeof(int), &tp);

  err = clEnqueueNDRangeKernel(queue, kernel_magnitude_reduction, 1, nullptr,
                                &global_size_reduction, &reduction_workgroup_size, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue compute_max_magnitude: %d\n", err);
    return;
  }

  int num_partials = (int)num_reduction_workgroups;
  size_t final_global_size = reduction_workgroup_size;

  clSetKernelArg(kernel_reduce_final, 0, sizeof(cl_mem), &d_buffer_partial_max);
  clSetKernelArg(kernel_reduce_final, 1, reduction_workgroup_size * sizeof(float), nullptr);
  clSetKernelArg(kernel_reduce_final, 2, sizeof(cl_mem), &d_buffer_final_max);
  clSetKernelArg(kernel_reduce_final, 3, sizeof(int), &num_partials);

  err = clEnqueueNDRangeKernel(queue, kernel_reduce_final, 1, NULL,
                                &final_global_size, &reduction_workgroup_size, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue reduce_max_final: %d\n", err);
    return;
  }

  err = clEnqueueReadBuffer(queue, d_buffer_final_max, CL_TRUE, 0,
                             sizeof(float), &global_max_mag_host, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to read max magnitude: %d\n", err);
    return;
  }

  int invert_int = invert_colors ? 1 : 0;
  cl_int use_log = use_log_scale ? 1 : 0;

  clSetKernelArg(kernel_render_frame, 0, sizeof(cl_mem), &d_buffer_output_frame);
  clSetKernelArg(kernel_render_frame, 1, sizeof(cl_mem), &d_buffer_real);
  clSetKernelArg(kernel_render_frame, 2, sizeof(cl_mem), &d_buffer_imag);
  clSetKernelArg(kernel_render_frame, 3, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_render_frame, 4, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_render_frame, 5, sizeof(float), &global_max_mag_host);
  clSetKernelArg(kernel_render_frame, 6, sizeof(float), &gamma_value);
  clSetKernelArg(kernel_render_frame, 7, sizeof(int), &invert_int);
  clSetKernelArg(kernel_render_frame, 8, sizeof(cl_int), &use_log);

  err = clEnqueueNDRangeKernel(queue, kernel_render_frame, 2, nullptr,
                                global_work_size_2d, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue render_frame: %d\n", err);
    return;
  }

  err = clEnqueueReadBuffer(queue, d_buffer_output_frame, CL_TRUE, 0,
                             total_pixels * 4 * sizeof(unsigned char),
                             host_pixels, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to read output frame: %d\n", err);
    return;
  }
}

static void render_point_cloud_gpu(complex long double *roots, const size_t *num_distinct, size_t num_perms, size_t stride, float point_radius,
                                   float scale, float x_off, float y_off) {
  if (!initialized || num_perms == 0 || !roots) return;

  cl_int err;

  // upload cleared host_pixels to GPU first
  err = clEnqueueWriteBuffer(queue, d_buffer_output_frame, CL_FALSE, 0,
                             total_pixels * 4 * sizeof(unsigned char),
                             host_pixels, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to upload cleared frame buffer\n");
    return;
  }

  // reallocate buffers if needed
  size_t total_roots = num_perms*stride;
  if (total_roots > d_perms_capacity) {
    if (d_perm_roots_real) clReleaseMemObject(d_perm_roots_real);
    if (d_perm_roots_imag) clReleaseMemObject(d_perm_roots_imag);

    d_perm_roots_real = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        total_roots * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "failed to create perm roots real buffer\n");
      return;
    }

    d_perm_roots_imag = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        total_roots * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "failed to create perm roots imag buffer\n");
      return;
    }

    d_perm_distinct = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        num_perms * sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "failed to create perm num distinct buffer\n");
      return;
    }

    d_perms_capacity = total_roots;
  }

  // upload root data
  cl_float *roots_real = malloc(total_roots * sizeof(cl_float));
  cl_float *roots_imag = malloc(total_roots * sizeof(cl_float));
  cl_int *distinct_roots = malloc(num_perms * sizeof(cl_int));

  for (size_t i = 0; i < total_roots; i++) {
    roots_real[i] = (cl_float)creall(roots[i]);
    roots_imag[i] = (cl_float)cimagl(roots[i]);
  }

  for (size_t i = 0; i < num_perms; i++) {
    distinct_roots[i] = (cl_int)num_distinct[i];
  }

  err = clEnqueueWriteBuffer(queue, d_perm_roots_real, CL_FALSE, 0,
                             total_roots * sizeof(cl_float), roots_real, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to write perm roots real buffer\n");
    free(roots_real);
    free(roots_imag);
    free(distinct_roots);
    return;
  }

  err = clEnqueueWriteBuffer(queue, d_perm_roots_imag, CL_FALSE, 0,
                             total_roots * sizeof(cl_float), roots_imag, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to write perm roots imag buffer\n");
    free(roots_real);
    free(roots_imag);
    free(distinct_roots);
    return;
  }

  err = clEnqueueWriteBuffer(queue, d_perm_distinct, CL_FALSE, 0,
                             num_perms * sizeof(cl_int), distinct_roots, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to write perm num distinct buffer\n");
    free(roots_real);
    free(roots_imag);
    free(distinct_roots);
    return;
  }

  clFinish(queue);

  free(roots_real);
  free(roots_imag);
  free(distinct_roots);

  // set kernel arguments
  cl_int nr = (int)num_perms;
  cl_int st = (int)stride;
  cl_int fw = (int)frame_width;
  cl_int fh = (int)frame_height;

  cl_float cl_scale = scale;
  cl_float cl_xoff = x_off;
  cl_float cl_yoff = y_off;
  cl_float cl_rad = point_radius;
  cl_float cl_sat = 0.1f + 0.9f * 32.0f / (float)num_perms;

  size_t global_work_size_2d[2] = {frame_width, frame_height};

  clSetKernelArg(kernel_clear_hue, 0, sizeof(cl_mem), &d_perm_roots_hue_ind);
  clSetKernelArg(kernel_clear_hue, 1, sizeof(cl_mem), &d_perm_roots_count);
  clSetKernelArg(kernel_clear_hue, 2, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_clear_hue, 3, sizeof(cl_int), &fh);

  err = clEnqueueNDRangeKernel(queue, kernel_clear_hue, 2, nullptr,
                                global_work_size_2d, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue clear_cloud_hue: %d\n", err);
    return;
  }

  clSetKernelArg(kernel_point_cloud, 0, sizeof(cl_mem), &d_perm_roots_hue_ind);
  clSetKernelArg(kernel_point_cloud, 1, sizeof(cl_mem), &d_perm_roots_count);
  clSetKernelArg(kernel_point_cloud, 2, sizeof(cl_mem), &d_perm_roots_real);
  clSetKernelArg(kernel_point_cloud, 3, sizeof(cl_mem), &d_perm_roots_imag);
  clSetKernelArg(kernel_point_cloud, 4, sizeof(cl_mem), &d_perm_distinct);
  clSetKernelArg(kernel_point_cloud, 5, sizeof(cl_int), &nr);
  clSetKernelArg(kernel_point_cloud, 6, sizeof(cl_int), &st);
  clSetKernelArg(kernel_point_cloud, 7, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_point_cloud, 8, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_point_cloud, 9, sizeof(cl_float), &cl_scale);
  clSetKernelArg(kernel_point_cloud, 10, sizeof(cl_float), &cl_xoff);
  clSetKernelArg(kernel_point_cloud, 11, sizeof(cl_float), &cl_yoff);
  clSetKernelArg(kernel_point_cloud, 12, sizeof(cl_float), &cl_rad);

  // launch kernels one work item per polynomial
  size_t global_work_size = num_perms;

  err = clEnqueueNDRangeKernel(queue, kernel_point_cloud, 1, nullptr,
                                &global_work_size, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to enqueue render_point_cloud: %d\n", err);
    return;
  }

  clSetKernelArg(kernel_point_cloud_hue, 0, sizeof(cl_mem), &d_buffer_output_frame);
  clSetKernelArg(kernel_point_cloud_hue, 1, sizeof(cl_mem), &d_perm_roots_hue_ind);
  clSetKernelArg(kernel_point_cloud_hue, 2, sizeof(cl_mem), &d_perm_roots_count);
  clSetKernelArg(kernel_point_cloud_hue, 3, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_point_cloud_hue, 4, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_point_cloud_hue, 5, sizeof(cl_float), &cl_sat);

  err = clEnqueueNDRangeKernel(queue, kernel_point_cloud_hue, 2, nullptr,
                                global_work_size_2d, nullptr, 0, nullptr, nullptr);

  // read back result
  err = clEnqueueReadBuffer(queue, d_buffer_output_frame, CL_TRUE, 0,
                             total_pixels * 4 * sizeof(unsigned char),
                             host_pixels, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "failed to read output frame after point cloud: %d\n", err);
    return;
  }
}
#endif

static void render_point_cloud_cpu(complex long double *roots, const size_t *num_distinct, size_t num_perms, size_t stride, float point_radius,
                                   float scale, float x_off, float y_off) {
  if (!host_pixels || num_perms == 0 || !roots) return;

  const int ir = (int)ceilf(point_radius);
  const float r2 = point_radius * point_radius;
  const float scale_saturation = 0.1f + 0.9f * 32.0f / (float)num_perms;

  for (int i = 0; i < total_pixels; i++) {
    hue_y_val[i] = 0.0f;
    hue_x_val[i] = 0.0f;
    hue_count[i] = 0;
  }
  for (size_t i = 0; i < num_perms; i++) {
    float hue = (float)i / (float)num_perms;
    for (size_t j = 0; j < num_distinct[i]; j++) {
      size_t idx = i * stride + j;
      float rx = (float)creall(roots[idx]);
      float ry = (float)cimagl(roots[idx]);

      // convert to screen coordinates
      float sx = (rx - x_off) / scale + (float)frame_width * 0.5f;
      float sy = (ry - y_off) / scale + (float)frame_height * 0.5f;

      // draw filled circle
      for (int dy = -ir; dy <= ir; dy++) {
        for (int dx = -ir; dx <= ir; dx++) {
          if ((float)(dx * dx + dy * dy) <= r2) {
            int px = (int)(sx + 0.5f) + dx;
            int py = (int)(sy + 0.5f) + dy;

            if (px >= 0 && px < (int)frame_width && py >= 0 && py < (int)frame_height) {
              size_t pix_idx = (size_t)py * frame_width + (size_t)px;

              hue_x_val[pix_idx] += cosf(2.0f * M_PI * hue);
              hue_y_val[pix_idx] += sinf(2.0f * M_PI * hue);
              hue_count[pix_idx] += 1;
            }

          }
        }
      }
    }
  }

  for (size_t i = 0; i < total_pixels; i++) {
    float mag = sqrtf(hue_x_val[i] * hue_x_val[i] + hue_y_val[i] * hue_y_val[i]);
    if (hue_count[i] > 0) {
      float angle = atanf(hue_y_val[i] / hue_x_val[i]);
      float hue = (angle + M_PI) / (2.0f * M_PI);

      float lightness = 0.7f;
      float saturation = 1.0f - (hue_count[i] - 1) * scale_saturation;
      if (saturation < 0.0f) {
        saturation = 0.0f;
      }

      unsigned char pr, pg, pb;
      hsl_to_rgb_cpu(hue, saturation, lightness, &pr, &pg, &pb);

      host_pixels[i * 4 + 0] = pr;
      host_pixels[i * 4 + 1] = pg;
      host_pixels[i * 4 + 2] = pb;
      host_pixels[i * 4 + 3] = 255;
    }
  }
}

void render_point_cloud(complex long double *roots, size_t *num_distinct, size_t num_perms, size_t stride, float point_radius,
                        float scale, float x_off, float y_off) {
#ifdef HAVE_OPENCL
  if (initialized) {
    render_point_cloud_gpu(roots, num_distinct, num_perms, stride, point_radius, scale, x_off, y_off);
  } else {
    render_point_cloud_cpu(roots, num_distinct, num_perms, stride, point_radius, scale, x_off, y_off);
  }
#else
  render_point_cloud_cpu(roots, num_distinct, num_perms, stride, point_radius, scale, x_off, y_off);
#endif
}

void render_frame_roots(float scale_inp, float x_off_inp, float y_off_inp) {
  if (!current_polynomial) return;

  pix_scale = scale_inp;
  x_offset = x_off_inp;
  y_offset = y_off_inp;

#ifdef HAVE_OPENCL
  if (initialized) {
    render_frame_gpu();
  } else {
    render_frame_cpu();
  }
#else
  render_frame_cpu();
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

const unsigned char* get_pixel_data(void) {
  return host_pixels;
}

float get_max_magnitude(void) {
  return global_max_mag_host;
}

const char* get_device_name(void) {
#ifdef HAVE_OPENCL
  if (initialized) {
    return device_name;
  }
#endif
  return "cpu (fallback)";
}