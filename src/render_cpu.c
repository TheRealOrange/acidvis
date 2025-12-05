//
// Created by Lin Yicheng on 30/11/25.
//

#include "render_internal.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>

// cpu rendering of polynomial evaluation heatmap
void render_frame_cpu(void) {
  if (!current_polynomial || !host_pixels) return;

  float max_mag = 0.0f;

  float *mags = malloc(total_pixels * sizeof(float));
  float *args = malloc(total_pixels * sizeof(float));

  if (!mags || !args) {
    free(mags);
    free(args);
    return;
  }

  for (size_t y = 0; y < frame_height; y++) {
    for (size_t x = 0; x < frame_width; x++) {
      float real_x = ((float)x - (float)frame_width * 0.5f) * pix_scale + x_offset;
      float real_y = ((float)y - (float)frame_height * 0.5f) * pix_scale + y_offset;
      cxldouble z = CXL(real_x, real_y);

      cxldouble result = polynomial_eval(current_polynomial, z);

      size_t idx = y * frame_width + x;
      float mag = (float)cxabsl(result);
      float arg = (float)cxargl(result);
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

// cpu rendering of point cloud
void render_point_cloud_cpu(cxldouble *roots, const bool *valid,
                            size_t num_perms, size_t stride, float point_radius,
                            float scale, float x_off, float y_off) {
  if (!host_pixels || num_perms == 0 || !roots) return;

  const int ir = (int)ceilf(point_radius);
  const float r2 = point_radius * point_radius;
  const float scale_saturation = 0.1f + 0.9f * 32.0f / (float)num_perms;

  // clear hue accumulators
  for (size_t i = 0; i < total_pixels; i++) {
    hue_y_val[i] = 0.0f;
    hue_x_val[i] = 0.0f;
    hue_count[i] = 0;
  }

  // accumulate hue values for each point
  for (size_t i = 0; i < num_perms; i++) {
    float hue = (float)i / (float)num_perms;
    if (!valid[i]) continue;
    for (size_t j = 0; j < stride; j++) {
      size_t idx = i * stride + j;
      float rx = (float)cxreall(roots[idx]);
      float ry = (float)cximagl(roots[idx]);

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

              hue_x_val[pix_idx] += cosf(2.0f * (float)M_PI * hue);
              hue_y_val[pix_idx] += sinf(2.0f * (float)M_PI * hue);
              hue_count[pix_idx] += 1;
            }
          }
        }
      }
    }
  }

  // convert hue accumulators to pixels
  for (size_t i = 0; i < total_pixels; i++) {
    if (hue_count[i] > 0) {
      float angle = atan2f(hue_y_val[i], hue_x_val[i]);
      float hue = (angle + (float)M_PI) / (2.0f * (float)M_PI);

      float lightness = 0.7f;
      float saturation = 1.0f - (float)(hue_count[i] - 1) * scale_saturation;
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
