// opencl kernel for polynomial visualization
// hue from argument, lightness from magnitude (white at roots, dark at max)

typedef float2 complexf;

// complex number operations

inline float cx_real(complexf a) {
  return a.x;
}

inline float cx_imag(complexf a) {
  return a.y;
}

inline complexf cx_conj(complexf a) {
  return (complexf)(a.x, -a.y);
}

inline float cx_mod(complexf a) {
  return hypot(a.x, a.y);
}

inline float cx_arg(complexf a) {
  return atan2(a.y, a.x);
}

inline complexf cx_mult(complexf a, complexf b) {
  return (complexf)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline complexf cx_add(complexf a, complexf b) {
  return (complexf)(a.x + b.x, a.y + b.y);
}

// hsl to rgb helper
inline float hue_to_rgb_component(float p, float q, float t) {
  if (t < 0.0f) t += 1.0f;
  if (t > 1.0f) t -= 1.0f;
  if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
  if (t < 0.5f) return q;
  if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  return p;
}

// h, s, l all in [0, 1], returns rgb in [0, 1]
inline float3 hsl_to_rgb(float h, float s, float l) {
  float r, g, b;

  if (s < 0.001f) {
    r = g = b = l;
  } else {
    float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    float p = 2.0f * l - q;
    r = hue_to_rgb_component(p, q, h + 1.0f / 3.0f);
    g = hue_to_rgb_component(p, q, h);
    b = hue_to_rgb_component(p, q, h - 1.0f / 3.0f);
  }

  return (float3)(r, g, b);
}

// evaluate polynomial at each pixel position
__kernel void evaluate_polynomial(
    __global float* out_real,
    __global float* out_imag,
    __global const float* coeffs_real,
    __global const float* coeffs_imag,
    const int num_coeffs,
    const int frame_width,
    const int frame_height,
    const float pixel_scale,
    const float offset_x,
    const float offset_y
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= frame_width || y >= frame_height) return;

  // compute complex coordinate centered on screen
  float real_x = ((float)x - (float)frame_width * 0.5f) * pixel_scale + offset_x;
  float real_y = ((float)y - (float)frame_height * 0.5f) * pixel_scale + offset_y;
  complexf z = (complexf)(real_x, real_y);

  // horner's method
  complexf result = (complexf)(0.0f, 0.0f);
  for (int i = num_coeffs - 1; i >= 0; i--) {
    complexf coeff = (complexf)(coeffs_real[i], coeffs_imag[i]);
    result = cx_add(cx_mult(result, z), coeff);
  }

  int idx = y * frame_width + x;
  out_real[idx] = result.x;
  out_imag[idx] = result.y;
}

// parallel reduction to find max magnitude
__kernel void compute_max_magnitude(
    __global const float* frame_real,
    __global const float* frame_imag,
    __local float* local_maxes,
    __global float* partial_max_out,
    const int total_pixels
) {
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);

  float mag = 0.0f;
  if (global_id < total_pixels) {
    float re = frame_real[global_id];
    float im = frame_imag[global_id];
    mag = hypot(re, im);
  }
  local_maxes[local_id] = mag;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = local_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      float other = local_maxes[local_id + stride];
      if (other > local_maxes[local_id]) {
        local_maxes[local_id] = other;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    partial_max_out[get_group_id(0)] = local_maxes[0];
  }
}

// final reduction across workgroups
__kernel void reduce_max_final(
    __global float* partial_maxes,
    __local float* local_maxes,
    __global float* final_max,
    const int num_partials
) {
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);

  // each work item finds max across its strided elements
  float val = 0.0f;
  for (int i = local_id; i < num_partials; i += local_size) {
    float v = partial_maxes[i];
    if (v > val) val = v;
  }

  local_maxes[local_id] = val;
  barrier(CLK_LOCAL_MEM_FENCE);

  // standard parallel reduction
  for (int stride = local_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      float other = local_maxes[local_id + stride];
      if (other > local_maxes[local_id]) {
        local_maxes[local_id] = other;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    final_max[0] = local_maxes[0];
  }
}

// render framebuffer with hsl coloring
__kernel void render_frame(
    __global uchar4* frame_out,
    __global const float* frame_real,
    __global const float* frame_imag,
    const int frame_width,
    const int frame_height,
    const float max_magnitude,
    const float gamma_value,
    const int invert_colors,
    const int use_log_scale
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= frame_width || y >= frame_height) return;

  int idx = y * frame_width + x;

  float re = frame_real[idx];
  float im = frame_imag[idx];

  float mag = hypot(re, im);
  float arg = atan2(im, re);

  // hue from argument [-pi, pi] -> [0, 1]
  float hue = (arg + M_PI_F) / (2.0f * M_PI_F);

  // lightness and saturation from magnitude
  float lightness;
  float saturation;
  if (max_magnitude < 1e-10f) {
    lightness = 1.0f;
    saturation = 0.0f;
  } else {
    float normalized;
    if (use_log_scale) {
      float log_mag = log1p(mag);
      float log_max = log1p(max_magnitude);
      normalized = clamp(log_mag / log_max, 0.0f, 1.0f);
    } else {
      normalized = clamp(mag / max_magnitude, 0.0f, 1.0f);
    }

    float high_contrast_gamma = pow(normalized, gamma_value);

    lightness = 1.0f - high_contrast_gamma * 0.93f;
    if (invert_colors) {
      lightness = 1.0f - lightness;
    }

    // low magnitude desaturated, high magnitude saturated
    saturation = 0.0f + high_contrast_gamma * 0.7f;
  }

  float3 rgb = hsl_to_rgb(hue, saturation, lightness);

  uchar4 pixel;
  pixel.x = (uchar)(rgb.x * 255.0f);
  pixel.y = (uchar)(rgb.y * 255.0f);
  pixel.z = (uchar)(rgb.z * 255.0f);
  pixel.w = 255;

  frame_out[idx] = pixel;
}

// Helper function for atomic float add using compare-and-swap
void atomicAdd_f(volatile __global float *addr, const float val) {
  union {
      unsigned int u;
      float f;
  } oldVal, newVal;
  int accepted = 0;

  // Read the current value as an integer bit pattern
  oldVal.f = *addr;

  do {
    // Calculate the new float value
    newVal.f = oldVal.f + val;

    // Attempt to swap: compare current address content (oldVal.u)
    // with the value we think it still holds. If they match,
    // swap in the new value (newVal.u).
    // This function returns the *actual* value that was in *addr* before the operation.
    oldVal.u = atomic_cmpxchg((volatile __global unsigned int *)addr, oldVal.u, newVal.u);

    // Loop continues if oldVal.u was different from what we expected,
    // meaning another thread modified the value in the meantime.
  } while (oldVal.u != newVal.u);
}

// render point cloud, each work item handles one polynomial's roots
__kernel void render_point_cloud(
    __global float* hue_ind,
    __global int* point_cnt,
    __global const float* roots_real,
    __global const float* roots_imag,
    __global const int* num_distinct,
    const int num_perms,
    const int stride,
    const int frame_width,
    const int frame_height,
    const float pixel_scale,
    const float offset_x,
    const float offset_y,
    const float point_radius
) {
  int perm_idx = get_global_id(0);
  if (perm_idx >= num_perms) return;

  int ir = (int)(point_radius + 0.5f);
  float r2 = point_radius * point_radius;

  for (int j = 0; j < num_distinct[perm_idx]; j++) {
    int root_idx = perm_idx * stride + j;

    float root_re = roots_real[root_idx];
    float root_im = roots_imag[root_idx];

    // convert complex coordinate to screen coordinate
    float sx = (root_re - offset_x) / pixel_scale + (float)frame_width * 0.5f;
    float sy = (root_im - offset_y) / pixel_scale + (float)frame_height * 0.5f;

    // skip if circle is entirely off-screen
    if (sx + ir < 0 || sx - ir >= frame_width ||
        sy + ir < 0 || sy - ir >= frame_height) {
      continue;
    }

    // draw filled circle
    float angle = 2.0f * (float) perm_idx / (float) num_perms;
    float cos_val = cospi(angle);
    float sin_val = sinpi(angle);
    for (int dy = -ir; dy <= ir; dy++) {
      for (int dx = -ir; dx <= ir; dx++) {
        if ((float)(dx * dx + dy * dy) <= r2) {
          int px = (int)(sx + 0.5f) + dx;
          int py = (int)(sy + 0.5f) + dy;

          if (px >= 0 && px < frame_width && py >= 0 && py < frame_height) {
            int idx = py * frame_width + px;

            atomicAdd_f(&hue_ind[idx * 2], cos_val);
            atomicAdd_f(&hue_ind[idx * 2 + 1], sin_val);
            atomic_add(&point_cnt[idx], 1);
          }
        }
      }
    }
  }
}

__kernel void clear_cloud_hue(
    __global float* hue_ind,
    __global int* point_cnt,
    const int frame_width,
    const int frame_height
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= frame_width || y >= frame_height) return;

  int idx = y * frame_width + x;
  hue_ind[idx * 2] = 0.0f;
  hue_ind[idx * 2 + 1] = 0.0f;
  point_cnt[idx] = 0;
}


__kernel void render_point_cloud_hue(
    __global uchar4* frame_out,
    __global const float* hue_ind,
    __global int* point_cnt,
    const int frame_width,
    const int frame_height,
    const float scale_sat
) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x >= frame_width || y >= frame_height) return;
  int idx = y * frame_width + x;

  if (point_cnt[idx] <= 0) {
    uchar4 black = (uchar4)(0, 0, 0, 255);
    frame_out[idx] = black;
    return;
  }

  float2 hue_vec = (float2)((float) hue_ind[idx * 2], (float) hue_ind[idx * 2 + 1]);
  float mag = length(hue_vec);
  float angle = atan2(hue_vec.y, hue_vec.x);

  float hue = (angle + M_PI_F) / (2.0f * M_PI_F);
  float lightness = 0.7f;
  float saturation = saturation = 1.0f - (point_cnt[idx] - 1) * scale_sat;

  if (saturation < 0.0f) {
    saturation = 0.0f;
  } else if (saturation > 1.0f) {
    saturation = 1.0f;
  }
  float3 rgb = hsl_to_rgb(hue, saturation, lightness);

  uchar4 pixel;
  pixel.x = (uchar)(rgb.x * 255.0f);
  pixel.y = (uchar)(rgb.y * 255.0f);
  pixel.z = (uchar)(rgb.z * 255.0f);
  pixel.w = 255;

  frame_out[idx] = pixel;
}