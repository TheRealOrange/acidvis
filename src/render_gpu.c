//
// Created by Lin Yicheng on 30/11/25.
//

#include "render_internal.h"

#ifdef HAVE_OPENCL

#include "generated/kernel_source.h"

#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// opencl state
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_program program = NULL;
static cl_device_id device = NULL;
static char device_name[256];

// kernels
static cl_kernel kernel_process = NULL;
static cl_kernel kernel_magnitude_reduction = NULL;
static cl_kernel kernel_reduce_final = NULL;
static cl_kernel kernel_render_frame = NULL;
static cl_kernel kernel_point_cloud = NULL;
static cl_kernel kernel_clear_hue = NULL;
static cl_kernel kernel_point_cloud_hue = NULL;

// device buffers
static cl_mem d_poly_coeff_real = NULL;
static cl_mem d_poly_coeff_imag = NULL;
static cl_mem d_buffer_real = NULL;
static cl_mem d_buffer_imag = NULL;
static cl_mem d_buffer_output_frame = NULL;
static cl_mem d_buffer_partial_max = NULL;
static cl_mem d_buffer_final_max = NULL;

// point cloud buffers
static cl_mem d_comb_roots_real = NULL;
static cl_mem d_comb_roots_imag = NULL;
static cl_mem d_comb_valid = NULL;
static cl_mem d_comb_roots_hue_ind = NULL;
static cl_mem d_comb_roots_count = NULL;
static size_t d_comb_capacity = 0;

// reduction workgroup settings
static size_t reduction_workgroup_size = 256;
static size_t num_reduction_workgroups = 0;

const char *render_gpu_get_device_name(void) {
  return device_name;
}

bool render_gpu_init(void) {
  cl_int err;
  cl_platform_id platform = NULL;

  SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "initializing opencl");

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to get opencl platform");
    return false;
  }

  // try gpu first, fall back to cpu
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to get opencl device");
      return false;
    }
  }

  err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  if (err != CL_SUCCESS) {
    strcpy(device_name, "unknown device");
  }

  SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "using opencl device: %s", device_name);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create opencl context");
    return false;
  }

#ifdef CL_VERSION_2_0
  cl_queue_properties properties[] = {0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
#else
  queue = clCreateCommandQueue(context, device, 0, &err);
#endif
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create command queue");
    return false;
  }

  // compile kernel
  const char *src_ptr = kernel_source;
  size_t src_len = strlen(kernel_source);
  program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create program");
    return false;
  }

  err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
  if (err != CL_SUCCESS) {
    char build_log[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(build_log), build_log, NULL);
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "opencl build error:\n%s", build_log);
    return false;
  }

  // create kernels
  kernel_process = clCreateKernel(program, "evaluate_polynomial", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create evaluate_polynomial kernel");
    return false;
  }

  kernel_magnitude_reduction = clCreateKernel(program, "compute_max_magnitude", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create compute_max_magnitude kernel");
    return false;
  }

  kernel_reduce_final = clCreateKernel(program, "reduce_max_final", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create reduce_max_final kernel");
    return false;
  }

  kernel_render_frame = clCreateKernel(program, "render_frame", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create render_frame kernel");
    return false;
  }

  kernel_point_cloud = clCreateKernel(program, "render_point_cloud", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create render_point_cloud kernel");
    return false;
  }

  kernel_clear_hue = clCreateKernel(program, "clear_cloud_hue", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create clear_cloud_hue kernel");
    return false;
  }

  kernel_point_cloud_hue = clCreateKernel(program, "render_point_cloud_hue", &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create render_point_cloud_hue kernel");
    return false;
  }

  d_buffer_final_max = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create final max buffer");
    return false;
  }

  gpu_initialized = true;
  SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "opencl initialized successfully");
  return true;
}

void render_gpu_cleanup(void) {
  SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "cleaning up opencl resources");

  if (kernel_process) clReleaseKernel(kernel_process);
  if (kernel_magnitude_reduction) clReleaseKernel(kernel_magnitude_reduction);
  if (kernel_reduce_final) clReleaseKernel(kernel_reduce_final);
  if (kernel_render_frame) clReleaseKernel(kernel_render_frame);
  if (kernel_point_cloud) clReleaseKernel(kernel_point_cloud);
  if (kernel_clear_hue) clReleaseKernel(kernel_clear_hue);
  if (kernel_point_cloud_hue) clReleaseKernel(kernel_point_cloud_hue);

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
  if (d_comb_roots_real) clReleaseMemObject(d_comb_roots_real);
  if (d_comb_roots_imag) clReleaseMemObject(d_comb_roots_imag);
  if (d_comb_valid) clReleaseMemObject(d_comb_valid);
  if (d_comb_roots_hue_ind) clReleaseMemObject(d_comb_roots_hue_ind);
  if (d_comb_roots_count) clReleaseMemObject(d_comb_roots_count);
}

void render_gpu_resize_buffers(int width, int height) {
  cl_int err;

  if (d_buffer_real) clReleaseMemObject(d_buffer_real);
  if (d_buffer_imag) clReleaseMemObject(d_buffer_imag);
  if (d_buffer_output_frame) clReleaseMemObject(d_buffer_output_frame);
  if (d_buffer_partial_max) clReleaseMemObject(d_buffer_partial_max);
  if (d_comb_roots_hue_ind) clReleaseMemObject(d_comb_roots_hue_ind);
  if (d_comb_roots_count) clReleaseMemObject(d_comb_roots_count);

  d_buffer_real = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 total_pixels * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create real buffer");
    return;
  }

  d_buffer_imag = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 total_pixels * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create imag buffer");
    return;
  }

  d_buffer_output_frame = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         total_pixels * 4 * sizeof(unsigned char), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create output frame buffer");
    return;
  }

  num_reduction_workgroups = (total_pixels + reduction_workgroup_size - 1) / reduction_workgroup_size;

  d_buffer_partial_max = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        num_reduction_workgroups * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create partial max buffer");
    return;
  }

  d_comb_roots_hue_ind = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        total_pixels * 2 * sizeof(cl_float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create hue index buffer");
    return;
  }

  d_comb_roots_count = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      total_pixels * sizeof(cl_int), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create hue count buffer");
    return;
  }

  SDL_LogDebug(SDL_LOG_CATEGORY_RENDER, "gpu buffers resized: %dx%d", width, height);
}

void render_gpu_set_polynomial(polynomial_t *P) {
  if (!gpu_initialized || !P) return;

  cl_int err;
  size_t num_coeffs = P->degree + 1;

  if (d_poly_coeff_real) clReleaseMemObject(d_poly_coeff_real);
  if (d_poly_coeff_imag) clReleaseMemObject(d_poly_coeff_imag);

  d_poly_coeff_real = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     num_coeffs * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create coefficient real buffer");
    return;
  }

  d_poly_coeff_imag = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     num_coeffs * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create coefficient imag buffer");
    return;
  }

  float *coeffs_real = malloc(num_coeffs * sizeof(float));
  float *coeffs_imag = malloc(num_coeffs * sizeof(float));

  for (size_t i = 0; i < num_coeffs; i++) {
    coeffs_real[i] = (float)cxreall(P->coeffs[i]);
    coeffs_imag[i] = (float)cximagl(P->coeffs[i]);
  }

  err = clEnqueueWriteBuffer(queue, d_poly_coeff_real, CL_FALSE, 0,
                             num_coeffs * sizeof(float), coeffs_real, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to write coefficient real buffer");
  }

  err = clEnqueueWriteBuffer(queue, d_poly_coeff_imag, CL_FALSE, 0,
                             num_coeffs * sizeof(float), coeffs_imag, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to write coefficient imag buffer");
  }

  clFinish(queue);

  free(coeffs_real);
  free(coeffs_imag);
}

void render_frame_gpu(void) {
  if (!current_polynomial || !gpu_initialized) return;

  cl_int err;

  size_t global_work_size_2d[2] = {frame_width, frame_height};
  cl_int num_coeffs = (cl_int)(current_polynomial->degree + 1);
  cl_int fw = (cl_int)frame_width;
  cl_int fh = (cl_int)frame_height;

  // evaluate polynomial at each pixel
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

  err = clEnqueueNDRangeKernel(queue, kernel_process, 2, NULL,
                               global_work_size_2d, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue evaluate_polynomial: %d", err);
    return;
  }

  // compute max magnitude via reduction
  size_t global_size_reduction = num_reduction_workgroups * reduction_workgroup_size;
  int tp = (int)total_pixels;

  clSetKernelArg(kernel_magnitude_reduction, 0, sizeof(cl_mem), &d_buffer_real);
  clSetKernelArg(kernel_magnitude_reduction, 1, sizeof(cl_mem), &d_buffer_imag);
  clSetKernelArg(kernel_magnitude_reduction, 2, reduction_workgroup_size * sizeof(float), NULL);
  clSetKernelArg(kernel_magnitude_reduction, 3, sizeof(cl_mem), &d_buffer_partial_max);
  clSetKernelArg(kernel_magnitude_reduction, 4, sizeof(int), &tp);

  err = clEnqueueNDRangeKernel(queue, kernel_magnitude_reduction, 1, NULL,
                               &global_size_reduction, &reduction_workgroup_size, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue compute_max_magnitude: %d", err);
    return;
  }

  // final reduction
  int num_partials = (int)num_reduction_workgroups;
  size_t final_global_size = reduction_workgroup_size;

  clSetKernelArg(kernel_reduce_final, 0, sizeof(cl_mem), &d_buffer_partial_max);
  clSetKernelArg(kernel_reduce_final, 1, reduction_workgroup_size * sizeof(float), NULL);
  clSetKernelArg(kernel_reduce_final, 2, sizeof(cl_mem), &d_buffer_final_max);
  clSetKernelArg(kernel_reduce_final, 3, sizeof(int), &num_partials);

  err = clEnqueueNDRangeKernel(queue, kernel_reduce_final, 1, NULL,
                               &final_global_size, &reduction_workgroup_size, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue reduce_max_final: %d", err);
    return;
  }

  // read back max magnitude
  err = clEnqueueReadBuffer(queue, d_buffer_final_max, CL_TRUE, 0,
                            sizeof(float), &global_max_mag_host, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to read max magnitude: %d", err);
    return;
  }

  // render frame with color mapping
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

  err = clEnqueueNDRangeKernel(queue, kernel_render_frame, 2, NULL,
                               global_work_size_2d, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue render_frame: %d", err);
    return;
  }

  // read back pixel data
  err = clEnqueueReadBuffer(queue, d_buffer_output_frame, CL_TRUE, 0,
                            total_pixels * 4 * sizeof(unsigned char),
                            host_pixels, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to read output frame: %d", err);
    return;
  }
}

void render_point_cloud_gpu(cxldouble *roots, const bool *valid,
                            size_t num_perms, size_t stride, float point_radius,
                            float scale, float x_off, float y_off) {
  if (!gpu_initialized || num_perms == 0 || !roots) return;

  cl_int err;

  // upload cleared host_pixels to gpu first
  err = clEnqueueWriteBuffer(queue, d_buffer_output_frame, CL_FALSE, 0,
                             total_pixels * 4 * sizeof(unsigned char),
                             host_pixels, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to upload cleared frame buffer");
    return;
  }

  // reallocate buffers if needed
  size_t total_roots = num_perms * stride;
  if (total_roots > d_comb_capacity) {
    if (d_comb_roots_real) clReleaseMemObject(d_comb_roots_real);
    if (d_comb_roots_imag) clReleaseMemObject(d_comb_roots_imag);
    if (d_comb_valid) clReleaseMemObject(d_comb_valid);

    d_comb_roots_real = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       total_roots * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) {
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create perm roots real buffer");
      return;
    }

    d_comb_roots_imag = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       total_roots * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) {
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create perm roots imag buffer");
      return;
    }

    d_comb_valid = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     num_perms * sizeof(cl_bool), NULL, &err);
    if (err != CL_SUCCESS) {
      SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to create perm comb_valid buffer");
      return;
    }

    d_comb_capacity = total_roots;
  }

  // upload root data
  cl_float *roots_real = malloc(total_roots * sizeof(cl_float));
  cl_float *roots_imag = malloc(total_roots * sizeof(cl_float));
  cl_bool *valid_roots = malloc(num_perms * sizeof(cl_bool));

  for (size_t i = 0; i < total_roots; i++) {
    roots_real[i] = (cl_float)cxreall(roots[i]);
    roots_imag[i] = (cl_float)cximagl(roots[i]);
  }

  for (size_t i = 0; i < num_perms; i++) {
    valid_roots[i] = (cl_bool)valid[i];
  }

  err = clEnqueueWriteBuffer(queue, d_comb_roots_real, CL_FALSE, 0,
                             total_roots * sizeof(cl_float), roots_real, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to write perm roots real buffer");
    goto cleanup;
  }

  err = clEnqueueWriteBuffer(queue, d_comb_roots_imag, CL_FALSE, 0,
                             total_roots * sizeof(cl_float), roots_imag, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to write perm roots imag buffer");
    goto cleanup;
  }

  err = clEnqueueWriteBuffer(queue, d_comb_valid, CL_FALSE, 0,
                             num_perms * sizeof(cl_bool), valid_roots, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to write perm num distinct buffer");
    goto cleanup;
  }

  clFinish(queue);

  // set kernel arguments
  cl_int nr = (cl_int)num_perms;
  cl_int st = (cl_int)stride;
  cl_int fw = (cl_int)frame_width;
  cl_int fh = (cl_int)frame_height;

  cl_float cl_scale = scale;
  cl_float cl_xoff = x_off;
  cl_float cl_yoff = y_off;
  cl_float cl_rad = point_radius;
  cl_float cl_sat = 0.1f + 0.9f * 32.0f / (float)num_perms;

  size_t global_work_size_2d[2] = {frame_width, frame_height};

  // clear hue accumulator
  clSetKernelArg(kernel_clear_hue, 0, sizeof(cl_mem), &d_comb_roots_hue_ind);
  clSetKernelArg(kernel_clear_hue, 1, sizeof(cl_mem), &d_comb_roots_count);
  clSetKernelArg(kernel_clear_hue, 2, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_clear_hue, 3, sizeof(cl_int), &fh);

  err = clEnqueueNDRangeKernel(queue, kernel_clear_hue, 2, NULL,
                               global_work_size_2d, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue clear_cloud_hue: %d", err);
    goto cleanup;
  }

  // render points to hue accumulator
  clSetKernelArg(kernel_point_cloud, 0, sizeof(cl_mem), &d_comb_roots_hue_ind);
  clSetKernelArg(kernel_point_cloud, 1, sizeof(cl_mem), &d_comb_roots_count);
  clSetKernelArg(kernel_point_cloud, 2, sizeof(cl_mem), &d_comb_roots_real);
  clSetKernelArg(kernel_point_cloud, 3, sizeof(cl_mem), &d_comb_roots_imag);
  clSetKernelArg(kernel_point_cloud, 4, sizeof(cl_mem), &d_comb_valid);
  clSetKernelArg(kernel_point_cloud, 5, sizeof(cl_int), &nr);
  clSetKernelArg(kernel_point_cloud, 6, sizeof(cl_int), &st);
  clSetKernelArg(kernel_point_cloud, 7, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_point_cloud, 8, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_point_cloud, 9, sizeof(cl_float), &cl_scale);
  clSetKernelArg(kernel_point_cloud, 10, sizeof(cl_float), &cl_xoff);
  clSetKernelArg(kernel_point_cloud, 11, sizeof(cl_float), &cl_yoff);
  clSetKernelArg(kernel_point_cloud, 12, sizeof(cl_float), &cl_rad);

  size_t global_work_size = num_perms;

  err = clEnqueueNDRangeKernel(queue, kernel_point_cloud, 1, NULL,
                               &global_work_size, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue render_point_cloud: %d", err);
    goto cleanup;
  }

  // convert hue accumulator to pixels
  clSetKernelArg(kernel_point_cloud_hue, 0, sizeof(cl_mem), &d_buffer_output_frame);
  clSetKernelArg(kernel_point_cloud_hue, 1, sizeof(cl_mem), &d_comb_roots_hue_ind);
  clSetKernelArg(kernel_point_cloud_hue, 2, sizeof(cl_mem), &d_comb_roots_count);
  clSetKernelArg(kernel_point_cloud_hue, 3, sizeof(cl_int), &fw);
  clSetKernelArg(kernel_point_cloud_hue, 4, sizeof(cl_int), &fh);
  clSetKernelArg(kernel_point_cloud_hue, 5, sizeof(cl_float), &cl_sat);

  err = clEnqueueNDRangeKernel(queue, kernel_point_cloud_hue, 2, NULL,
                               global_work_size_2d, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to enqueue render_point_cloud_hue: %d", err);
    goto cleanup;
  }

  // read back result
  err = clEnqueueReadBuffer(queue, d_buffer_output_frame, CL_TRUE, 0,
                            total_pixels * 4 * sizeof(unsigned char),
                            host_pixels, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "failed to read output frame after point cloud: %d", err);
  }

cleanup:
  free(roots_real);
  free(roots_imag);
  free(valid_roots);
}

#endif // HAVE_OPENCL
