//
// Created by Lin Yicheng on 30/11/25.
//

#include "ui.h"
#include "app.h"
#include "input.h"
#include "render.h"
#include "util.h"

#include <SDL3/SDL.h>
#include <math.h>

// draw a filled circle using scanline fill
void ui_draw_circle(SDL_Renderer *ren, float cx, float cy, float r) {
  int ir = (int)ceilf(r);
  for (int dy = -ir; dy <= ir; dy++) {
    float dx = sqrtf(r * r - (float)(dy * dy));
    SDL_RenderLine(ren, cx - dx, cy + (float)dy, cx + dx, cy + (float)dy);
  }
}

// draw a circle outline
void ui_draw_circle_outline(SDL_Renderer *ren, float cx, float cy, float r, int segments) {
  float prev_x = cx + r;
  float prev_y = cy;
  for (int i = 1; i <= segments; i++) {
    float angle = (float)i / (float)segments * 2.0f * (float)M_PI;
    float x = cx + r * cosf(angle);
    float y = cy + r * sinf(angle);
    SDL_RenderLine(ren, prev_x, prev_y, x, y);
    prev_x = x;
    prev_y = y;
  }
}

// render the info overlay
void ui_render_info_overlay(AppState *state) {
  SDL_SetRenderDrawColor(state->ren, 255, 255, 255, 255);

  if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
    // point cloud mode info
    size_t displayed_combs = state->drag_skip > 1 ?
        (state->num_combinations + state->drag_skip - 1) / state->drag_skip :
        state->num_combinations;

    SDL_RenderDebugTextFormat(state->ren, 5, 5,
        "view: (%.3f, %.3f) scale: %.5f  radius: %.2f  points: %s",
        state->center_x, state->center_y, state->scale,
        state->cloud_point_radius, state->fixed_point_scale ? "fixed" : "zoom");

#ifdef HAVE_LAPACK
    SDL_RenderDebugTextFormat(state->ren, 5, 20,
        "coeffs: %zu  degree: %zu  combinations: %zu/%zu  roots: %zu  solver: %s",
        state->num_base_coeffs, state->poly_degree_cloud,
        displayed_combs, state->num_combinations, state->found_roots,
        state->lapack ? "lapack" : "jt");
#else
    SDL_RenderDebugTextFormat(state->ren, 5, 20,
        "coeffs: %zu  degree: %zu  combinations: %zu/%zu  roots: %zu",
        state->num_base_coeffs, state->poly_degree_cloud,
        displayed_combs, state->num_combinations, state->found_roots);
#endif
  } else {
    // roots mode info
    SDL_RenderDebugTextFormat(state->ren, 5, 5,
        "view: (%.3f, %.3f) scale: %.5f  gamma: %.2f  invert: %s  log: %s",
        state->center_x, state->center_y, state->scale, get_gamma(),
        get_invert() ? "on" : "off", is_log_scale() ? "on" : "off");

#ifdef HAVE_LAPACK
    SDL_RenderDebugTextFormat(state->ren, 5, 20,
        "degree: %zu  roots: %zu  solver: %s",
        state->poly ? state->poly->degree : 0,
        state->poly ? state->poly->num_distinct_roots : 0,
        state->lapack ? "lapack" : "jt");
#else
    SDL_RenderDebugTextFormat(state->ren, 5, 20,
        "degree: %zu  roots: %zu",
        state->poly ? state->poly->degree : 0,
        state->poly ? state->poly->num_distinct_roots : 0);
#endif
  }

  // get keycodes for current keyboard layout
  SDL_Keycode r_key = SDL_GetKeyFromScancode(SCANCODE_RESET_VIEW, SDL_KMOD_NONE, true);
  SDL_Keycode c_key = SDL_GetKeyFromScancode(SCANCODE_RESET_POLY, SDL_KMOD_NONE, true);
  SDL_Keycode y_key = SDL_GetKeyFromScancode(SCANCODE_INC_DEGREE, SDL_KMOD_NONE, true);
  SDL_Keycode h_key = SDL_GetKeyFromScancode(SCANCODE_DEC_DEGREE, SDL_KMOD_NONE, true);
  SDL_Keycode add_key = SDL_GetKeyFromScancode(SCANCODE_ADD_COEFF, SDL_KMOD_NONE, true);
  SDL_Keycode rem_key = SDL_GetKeyFromScancode(SCANCODE_REM_COEFF, SDL_KMOD_NONE, true);
  SDL_Keycode n_key = SDL_GetKeyFromScancode(SCANCODE_INC_GAMMA, SDL_KMOD_NONE, true);
  SDL_Keycode m_key = SDL_GetKeyFromScancode(SCANCODE_DEC_GAMMA, SDL_KMOD_NONE, true);
  SDL_Keycode t_key = SDL_GetKeyFromScancode(SCANCODE_TOGGLE_MODE, SDL_KMOD_NONE, true);
  SDL_Keycode v_key = SDL_GetKeyFromScancode(SCANCODE_INVERT, SDL_KMOD_NONE, true);
  SDL_Keycode s_key = SDL_GetKeyFromScancode(SCANCODE_SCALE_MODE, SDL_KMOD_NONE, true);
  SDL_Keycode i_key = SDL_GetKeyFromScancode(SCANCODE_PAN_UP, SDL_KMOD_NONE, true);
  SDL_Keycode j_key = SDL_GetKeyFromScancode(SCANCODE_PAN_LEFT, SDL_KMOD_NONE, true);
  SDL_Keycode k_key = SDL_GetKeyFromScancode(SCANCODE_PAN_DOWN, SDL_KMOD_NONE, true);
  SDL_Keycode l_key = SDL_GetKeyFromScancode(SCANCODE_PAN_RIGHT, SDL_KMOD_NONE, true);
  SDL_Keycode u_key = SDL_GetKeyFromScancode(SCANCODE_INC_ZOOM, SDL_KMOD_NONE, true);
  SDL_Keycode o_key = SDL_GetKeyFromScancode(SCANCODE_DEC_ZOOM, SDL_KMOD_NONE, true);
  SDL_Keycode x_key = SDL_GetKeyFromScancode(SCANCODE_ANIM_RESET, SDL_KMOD_NONE, true);
  SDL_Keycode z_key = SDL_GetKeyFromScancode(SCANCODE_LOG_SCALE, SDL_KMOD_NONE, true);
  SDL_Keycode q_key = SDL_GetKeyFromScancode(SCANCODE_TOGGLE_ROOTS, SDL_KMOD_NONE, true);
  SDL_Keycode w_key = SDL_GetKeyFromScancode(SCANCODE_TOGGLE_COEFFS, SDL_KMOD_NONE, true);
#ifdef HAVE_LAPACK
  SDL_Keycode b_key = SDL_GetKeyFromScancode(SCANCODE_LAPACK, SDL_KMOD_NONE, true);
#endif

  // mode specific keys
  if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
    if (state->anim_active) {
      SDL_RenderDebugTextFormat(state->ren, 5, 40,
          "[%c/%c] radius  [%c] scale mode  [%c] show coeffs",
          n_key, m_key, s_key, w_key);
    } else {
      SDL_RenderDebugTextFormat(state->ren, 5, 40,
          "[%c/%c] degree  [%c/%c] coeffs  [%c/%c] radius  [%c] scale mode  [%c] show coeffs  [%c] roots mode",
          y_key, h_key, add_key, rem_key, n_key, m_key, s_key, w_key, t_key);
    }
  } else {
    if (state->anim_active) {
      SDL_RenderDebugTextFormat(state->ren, 5, 40,
          "[%c/%c] gamma  [%c] invert  [%c] log scale  [%c] show roots  [%c] show coeffs",
          n_key, m_key, v_key, z_key, q_key, w_key);
    } else {
      SDL_RenderDebugTextFormat(state->ren, 5, 40,
          "[%c/%c] degree  [%c/%c] gamma  [%c] inv  [%c] log  [%c] roots  [%c] coeffs  [%c] cloud",
          y_key, h_key, n_key, m_key, v_key, z_key, q_key, w_key, t_key);
    }
  }

  // common controls
  if (state->anim_active) {
    SDL_RenderDebugTextFormat(state->ren, 5, 55,
        "[space] play/pause  [%c] reset  [%c] reset view [%c%c%c%c] pan  [%c/%c] zoom  [F1] hide  [esc] quit",
        x_key, r_key, i_key, j_key, k_key, l_key, u_key, o_key);
  } else {
#ifdef HAVE_LAPACK
    SDL_RenderDebugTextFormat(state->ren, 5, 55,
        "[%c] reset view  [%c] reset poly  [%c%c%c%c] pan  [%c/%c] zoom  [%c] solver  [F1] hide  [esc] quit",
        r_key, c_key, i_key, j_key, k_key, l_key, u_key, o_key, b_key);
#else
    SDL_RenderDebugTextFormat(state->ren, 5, 55,
        "[%c] reset view  [%c] reset poly  [%c%c%c%c] pan  [%c/%c] zoom  [F1] hide  [esc] quit",
        r_key, c_key, i_key, j_key, k_key, l_key, u_key, o_key);
#endif
  }

  // animation status
  if (state->anim_active && state->anim_state) {
    const char *status = state->anim_state->playing ? "playing" :
                        (state->anim_state->finished ? "finished" : "paused");

    SDL_RenderDebugTextFormat(state->ren, 5, 70,
        "anim: %.1f/%.1fs %s%s",
        state->anim_state->current_time,
        state->anim_script->total_duration,
        status,
        state->anim_script->loop ? " [loop]" : "");
  }

  // show solve stats during animation or coefficient dragging in cloud mode
  bool show_solve_stats = (state->anim_active && state->view_mode == VIEW_MODE_POINT_CLOUD) ||
                          (state->drag_mode == DRAG_COEFF && state->view_mode == VIEW_MODE_POINT_CLOUD);
  if (show_solve_stats) {
    int y_pos = (state->anim_active && state->anim_state) ? 85 : 70;
    SDL_RenderDebugTextFormat(state->ren, 5, y_pos,
        "solver: incr %zu  full %zu  (failed %zu)",
        state->solve_stats.num_incremental,
        state->solve_stats.num_fullsolve,
        state->solve_stats.num_incremental_failed);
  }
}

// render coefficient/root point markers
void ui_render_point_markers(AppState *state) {
  float hue;
  uint8_t r, g, b;

  // draw coefficients or base_coeffs depending on mode
  if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->base_coeffs && state->show_coeffs) {
    // cloud mode: draw base coefficients (the N draggable points)
    for (size_t i = 0; i < state->num_base_coeffs && i < 26; i++) {
      float cx = (float)cxreall(state->base_coeffs[i]);
      float cy = (float)cximagl(state->base_coeffs[i]);
      float sx, sy;
      complex_to_screen(state, cx, cy, &sx, &sy);

      bool is_hover = (state->hover_mode == DRAG_COEFF && state->hover_index == (int)i);
      bool is_drag = (state->drag_mode == DRAG_COEFF && state->drag_index == (int)i);

      hue = (float)i / (float)state->num_base_coeffs;
      if (is_hover || is_drag) {
        hsl_to_rgb_cpu(hue, 0.8f, 0.7f, &r, &g, &b);
      } else {
        hsl_to_rgb_cpu(hue, 0.8f, 0.5f, &r, &g, &b);
      }
      SDL_SetRenderDrawColor(state->ren, r, g, b, 255);

      float rad = is_hover || is_drag ? POINT_RADIUS + 2 : POINT_RADIUS;
      ui_draw_circle(state->ren, sx, sy, rad);

      // outline
      SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
      ui_draw_circle_outline(state->ren, sx, sy, rad, 24);

      // draw label (a-z)
      char label[2] = { 'a' + (char)i, '\0' };
      SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
      SDL_RenderDebugText(state->ren, sx - 4, sy - 4, label);
    }
  } else if (state->poly && state->poly->coeffs_valid && state->show_coeffs) {
    // roots mode: draw polynomial coefficients
    for (size_t i = 0; i <= state->poly->degree && i < 26; i++) {
      float cx = (float)cxreall(state->poly->coeffs[i]);
      float cy = (float)cximagl(state->poly->coeffs[i]);
      float sx, sy;
      complex_to_screen(state, cx, cy, &sx, &sy);

      bool is_hover = (state->hover_mode == DRAG_COEFF && state->hover_index == (int)i);
      bool is_drag = (state->drag_mode == DRAG_COEFF && state->drag_index == (int)i);

      hue = (float)i / (float)(state->poly->degree + 1);
      if (is_hover || is_drag) {
        hsl_to_rgb_cpu(hue, 0.8f, 0.7f, &r, &g, &b);
      } else {
        hsl_to_rgb_cpu(hue, 0.8f, 0.5f, &r, &g, &b);
      }
      SDL_SetRenderDrawColor(state->ren, r, g, b, 255);

      float rad = is_hover || is_drag ? POINT_RADIUS + 2 : POINT_RADIUS;
      ui_draw_circle(state->ren, sx, sy, rad);

      // outline
      SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
      ui_draw_circle_outline(state->ren, sx, sy, rad, 24);

      // draw label (a-z)
      char label[2] = { 'a' + (char)i, '\0' };
      SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
      SDL_RenderDebugText(state->ren, sx - 4, sy - 4, label);
    }
  }

  // draw roots (only in roots mode)
  if (state->poly && state->poly->roots_valid) {
    if (state->view_mode == VIEW_MODE_ROOTS && state->show_roots) {
      for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
        float rx = (float)cxreall(state->poly->roots[i]);
        float ry = (float)cximagl(state->poly->roots[i]);
        float sx, sy;
        complex_to_screen(state, rx, ry, &sx, &sy);

        bool is_hover = (state->hover_mode == DRAG_ROOT && state->hover_index == (int)i);
        bool is_drag = (state->drag_mode == DRAG_ROOT && state->drag_index == (int)i);

        if (is_hover || is_drag) {
          SDL_SetRenderDrawColor(state->ren, 100, 100, 255, 255);
        } else {
          SDL_SetRenderDrawColor(state->ren, 255, 255, 255, 255);
        }

        float rad = is_hover || is_drag ? POINT_RADIUS + 2 : POINT_RADIUS;
        ui_draw_circle(state->ren, sx, sy, rad);

        // outline
        SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
        ui_draw_circle_outline(state->ren, sx, sy, rad, 24);
      }
    }
  }
}

// render the complete frame
void ui_render_frame(AppState *state) {
  // render the polynomial visualization to the pixel buffer
  if (state->view_mode == VIEW_MODE_ROOTS) {
    render_frame_roots(state->scale, state->center_x, state->center_y);
  } else {
    // clear to dark background
    clear_frame_buffer(20, 20, 25);

    if (state->combination_roots && state->num_combinations > 0) {
      // calculate base radius from user setting, scaled by number of combinations
      float n = (float)state->num_combinations;
      float scale_factor = 22.0f / powf(n + 1.0f, 0.3f) - 0.6f;
      scale_factor = fmaxf(0.5f, fminf(6.5f, scale_factor));
      float radius = state->cloud_point_radius * scale_factor;

      // if not fixed scale, adjust radius with zoom level
      if (!state->fixed_point_scale) {
        float zoom_factor = 0.005f / state->scale;  // 0.005 is default scale
        radius *= sqrtf(zoom_factor);
      }

      render_point_cloud(state->combination_roots, state->combination_valid,
                         state->num_combinations, state->comb_roots_stride, radius,
                         state->scale, state->center_x, state->center_y);
    }
  }

  // copy pixel data to texture
  const unsigned char *pixels = get_pixel_data();
  if (pixels) {
    SDL_UpdateTexture(state->texture, NULL, pixels, state->width * 4);
  }

  // clear and draw
  SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
  SDL_RenderClear(state->ren);
  SDL_RenderTexture(state->ren, state->texture, NULL, NULL);

  // draw point markers
  ui_render_point_markers(state);

  // info overlay
  if (state->show_info) {
    ui_render_info_overlay(state);
  } else {
    SDL_SetRenderDrawColor(state->ren, 128, 128, 128, 255);
    SDL_RenderDebugTextFormat(state->ren, 5, 5, "[F1] show info");
  }

  SDL_RenderPresent(state->ren);
  state->needs_redraw = false;
}
