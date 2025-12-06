//
// Created by Lin Yicheng on 30/11/25.
//

#include "input.h"
#include "app.h"
#include "cloud.h"
#include "render.h"
#include "ui.h"
#include "util.h"

#include <SDL3/SDL.h>
#include <math.h>
#include <stdlib.h>

// coordinate conversion helpers

void screen_to_complex(AppState *state, float sx, float sy, float *cx, float *cy) {
  *cx = (sx - state->width * 0.5f) * state->scale + state->center_x;
  *cy = (sy - state->height * 0.5f) * state->scale + state->center_y;
}

void complex_to_screen(AppState *state, float cx, float cy, float *sx, float *sy) {
  *sx = (cx - state->center_x) / state->scale + state->width * 0.5f;
  *sy = (cy - state->center_y) / state->scale + state->height * 0.5f;
}

// check if screen point is near a complex point
bool input_point_hit_test(AppState *state, float mouse_x, float mouse_y,
                          float complex_x, float complex_y) {
  float sx, sy;
  complex_to_screen(state, complex_x, complex_y, &sx, &sy);
  float dx = mouse_x - sx;
  float dy = mouse_y - sy;
  return (dx * dx + dy * dy) <= POINT_HIT_RADIUS * POINT_HIT_RADIUS;
}

// find what point (if any) is under the mouse
void input_find_hover(AppState *state, float mouse_x, float mouse_y) {
  state->hover_mode = DRAG_NONE;
  state->hover_index = -1;

  if (!state->poly) return;

  // check roots (only in roots mode)
  if (state->view_mode == VIEW_MODE_ROOTS && state->poly->roots_valid) {
    for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
      float rx = (float)cxreall(state->poly->roots[i]);
      float ry = (float)cximagl(state->poly->roots[i]);
      if (input_point_hit_test(state, mouse_x, mouse_y, rx, ry)) {
        state->hover_mode = DRAG_ROOT;
        state->hover_index = (int)i;
        return;
      }
    }
  }

  // in cloud mode check base_coeffs
  if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->base_coeffs && state->show_coeffs) {
    for (size_t i = 0; i < state->num_base_coeffs; i++) {
      float cx = (float)cxreall(state->base_coeffs[i]);
      float cy = (float)cximagl(state->base_coeffs[i]);
      if (input_point_hit_test(state, mouse_x, mouse_y, cx, cy)) {
        state->hover_mode = DRAG_COEFF;
        state->hover_index = (int)i;
        return;
      }
    }
  } else if (state->poly->coeffs_valid && state->show_coeffs) {
    for (size_t i = 0; i <= state->poly->degree; i++) {
      float cx = (float)cxreall(state->poly->coeffs[i]);
      float cy = (float)cximagl(state->poly->coeffs[i]);
      if (input_point_hit_test(state, mouse_x, mouse_y, cx, cy)) {
        state->hover_mode = DRAG_COEFF;
        state->hover_index = (int)i;
        return;
      }
    }
  }
}

// handle keyboard press events
static SDL_AppResult handle_key_down(AppState *state, SDL_Event *event) {
  size_t curr_deg = state->poly->degree;

  switch (event->key.scancode) {
    case SCANCODE_RESET_VIEW:
      state->scale = 0.005f;
      state->center_x = 0.0f;
      state->center_y = 0.0f;
      state->needs_redraw = true;
      break;

    case SCANCODE_RESET_POLY:
      if (!app_editing_allowed(state)) break;
      polynomial_free(state->poly);
      state->poly = app_create_default_polynomial(curr_deg);
      set_polynomial(state->poly);
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        if (state->base_coeffs) {
          for (size_t i = 0; i < state->num_base_coeffs; i++) {
            float angle = (float)i * 2.0f * (float)M_PI / (float)state->num_base_coeffs;
            state->base_coeffs[i] = cxexpl(cxscalel(CXL_I, angle));
          }
        }
        cloud_update(state, 1);
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_INC_DEGREE:
      if (!app_editing_allowed(state)) break;
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        size_t new_deg = state->poly_degree_cloud + 1;
        if (int_pow(state->num_base_coeffs, new_deg + 1) > 1000000) {
          SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "too many combinations");
          break;
        }
        state->poly_degree_cloud = new_deg;
        cloud_reallocate_combinations(state);
        cloud_update(state, 1);
      } else {
        curr_deg += (curr_deg < MAX_DEGREE) ? 1 : 0;
        polynomial_free(state->poly);
        state->poly = app_create_default_polynomial(curr_deg);
        set_polynomial(state->poly);
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_DEC_DEGREE:
      if (!app_editing_allowed(state)) break;
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        if (state->poly_degree_cloud > 2) {
          state->poly_degree_cloud--;
          cloud_reallocate_combinations(state);
          cloud_update(state, 1);
        }
      } else {
        curr_deg -= (curr_deg > 2) ? 1 : 0;
        polynomial_free(state->poly);
        state->poly = app_create_default_polynomial(curr_deg);
        set_polynomial(state->poly);
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_ADD_COEFF:
      if (!app_editing_allowed(state)) break;
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        size_t new_count = state->num_base_coeffs + 1;
        if (int_pow(new_count, state->poly_degree_cloud + 1) > 1000000) {
          SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "too many combinations");
          break;
        }
        cxldouble *new_coeffs = realloc(state->base_coeffs,
                                                   new_count * sizeof(cxldouble));
        if (new_coeffs) {
          state->base_coeffs = new_coeffs;
          state->base_coeffs[state->num_base_coeffs] = CXL(1.0, 0.0);
          state->num_base_coeffs = new_count;
          cloud_reallocate_combinations(state);
          cloud_update(state, 1);
          state->needs_redraw = true;
        }
      }
      break;

    case SCANCODE_REM_COEFF:
      if (!app_editing_allowed(state)) break;
      if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->num_base_coeffs > 2) {
        state->num_base_coeffs--;
        cloud_reallocate_combinations(state);
        cloud_update(state, 1);
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_TOGGLE_MODE:
      if (!app_editing_allowed(state)) break;
      if (state->view_mode == VIEW_MODE_ROOTS) {
        cloud_mode_enter(state);
      } else {
        cloud_mode_exit(state);
        state->view_mode = VIEW_MODE_ROOTS;
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_LAPACK:
      if (!app_editing_allowed(state)) break;
      state->lapack = !state->lapack;
      SDL_LogInfo(SDL_LOG_CATEGORY_INPUT, "solver: %s", state->lapack ? "lapack" : "jenkins-traub");
      state->needs_redraw = true;
      break;

    case SCANCODE_INVERT:
      if (state->view_mode == VIEW_MODE_ROOTS) {
        toggle_invert();
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_LOG_SCALE:
      if (state->view_mode == VIEW_MODE_ROOTS) {
        toggle_log_scale();
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_TOGGLE_ROOTS:
      if (state->view_mode == VIEW_MODE_ROOTS) {
        state->show_roots = !state->show_roots;
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_TOGGLE_COEFFS:
      state->show_coeffs = !state->show_coeffs;
      state->needs_redraw = true;
      break;

    case SCANCODE_TOGGLE_INFO:
      state->show_info = !state->show_info;
      state->needs_redraw = true;
      break;

    case SCANCODE_SCALE_MODE:
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        state->fixed_point_scale = !state->fixed_point_scale;
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_PAN_UP:    state->pan_keys[0] = true; break;
    case SCANCODE_PAN_LEFT:  state->pan_keys[1] = true; break;
    case SCANCODE_PAN_DOWN:  state->pan_keys[2] = true; break;
    case SCANCODE_PAN_RIGHT: state->pan_keys[3] = true; break;
    case SCANCODE_INC_ZOOM:  state->zoom_keys[0] = true; break;
    case SCANCODE_DEC_ZOOM:  state->zoom_keys[1] = true; break;

    case SCANCODE_INC_GAMMA:
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        state->cloud_point_radius += 0.2f;
        if (state->cloud_point_radius > CLOUD_POINT_RADIUS_MAX) {
          state->cloud_point_radius = CLOUD_POINT_RADIUS_MAX;
        }
      } else {
        float g = get_gamma();
        set_gamma(g + (g < 3.0f ? 0.05f : 0.0f));
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_DEC_GAMMA:
      if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
        state->cloud_point_radius -= 0.2f;
        if (state->cloud_point_radius < CLOUD_POINT_RADIUS_MIN) {
          state->cloud_point_radius = CLOUD_POINT_RADIUS_MIN;
        }
      } else {
        float g = get_gamma();
        set_gamma(g - (g > 0.1f ? 0.05f : 0.0f));
      }
      state->needs_redraw = true;
      break;

    case SCANCODE_ANIM_PLAY:
      if (state->anim_active && state->anim_state) {
        if (state->anim_state->finished) {
          anim_reset(state->anim_state);
          app_apply_animation_frame(state);
        }
        state->anim_state->playing = !state->anim_state->playing;
        state->anim_last_tick = SDL_GetTicks();
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_ANIM_RESET:
      if (state->anim_active && state->anim_state) {
        anim_reset(state->anim_state);
        app_apply_animation_frame(state);
        state->anim_last_tick = SDL_GetTicks();
        state->needs_redraw = true;
      }
      break;

    case SCANCODE_QUIT:
      return SDL_APP_SUCCESS;

    default:
      break;
  }

  return SDL_APP_CONTINUE;
}

// handle keyboard release events
static void handle_key_up(AppState *state, SDL_Event *event) {
  switch (event->key.scancode) {
    case SCANCODE_PAN_UP:    state->pan_keys[0] = false; break;
    case SCANCODE_PAN_LEFT:  state->pan_keys[1] = false; break;
    case SCANCODE_PAN_DOWN:  state->pan_keys[2] = false; break;
    case SCANCODE_PAN_RIGHT: state->pan_keys[3] = false; break;
    case SCANCODE_INC_ZOOM:  state->zoom_keys[0] = false; break;
    case SCANCODE_DEC_ZOOM:  state->zoom_keys[1] = false; break;
    default: break;
  }
}

// handle mouse button press
static void handle_mouse_down(AppState *state, SDL_Event *event) {
  float mx = event->button.x;
  float my = event->button.y;

  if (state->drag_mode == DRAG_NONE && event->button.button == SDL_BUTTON_LEFT) {
    input_find_hover(state, mx, my);

    if (app_editing_allowed(state)) {
      if (state->hover_mode == DRAG_ROOT) {
        state->drag_mode = DRAG_ROOT;
        state->drag_index = state->hover_index;
      } else if (state->hover_mode == DRAG_COEFF) {
        state->drag_mode = DRAG_COEFF;
        state->drag_index = state->hover_index;
      } else {
        // no point hit, pan
        state->drag_mode = DRAG_PAN;
        state->drag_start_mouse_x = mx;
        state->drag_start_mouse_y = my;
        state->drag_start_center_x = state->center_x;
        state->drag_start_center_y = state->center_y;
      }
    } else {
      // animation active, always pan
      state->drag_mode = DRAG_PAN;
      state->drag_start_mouse_x = mx;
      state->drag_start_mouse_y = my;
      state->drag_start_center_x = state->center_x;
      state->drag_start_center_y = state->center_y;
    }
  } else if (event->button.button == SDL_BUTTON_RIGHT) {
    // right click always pans
    state->drag_mode = DRAG_PAN;
    state->drag_start_mouse_x = mx;
    state->drag_start_mouse_y = my;
    state->drag_start_center_x = state->center_x;
    state->drag_start_center_y = state->center_y;
  }
}

// handle mouse button release
static void handle_mouse_up(AppState *state, SDL_Event *event) {
  if (event->button.button == SDL_BUTTON_LEFT ||
      event->button.button == SDL_BUTTON_RIGHT) {
    bool was_dragging_coeff_cloud = (state->drag_mode == DRAG_COEFF &&
                                     state->view_mode == VIEW_MODE_POINT_CLOUD);
    state->drag_mode = DRAG_NONE;
    state->drag_index = -1;

    // full recalculation after drag ends
    if (was_dragging_coeff_cloud) {
      state->drag_skip = 1;
      cloud_update_drag_end(state);
      state->needs_redraw = true;
    }
  }
}

// handle mouse motion
static void handle_mouse_motion(AppState *state, SDL_Event *event) {
  float mx = event->motion.x;
  float my = event->motion.y;

  if (state->drag_mode == DRAG_PAN) {
    float dx = (mx - state->drag_start_mouse_x) * state->scale;
    float dy = (my - state->drag_start_mouse_y) * state->scale;
    state->center_x = state->drag_start_center_x - dx;
    state->center_y = state->drag_start_center_y - dy;
    state->needs_redraw = true;

  } else if (state->drag_mode == DRAG_ROOT && state->drag_index >= 0) {
    float cx, cy;
    screen_to_complex(state, mx, my, &cx, &cy);
    state->poly->roots[state->drag_index] = CXL(cx, cy);
    app_rebuild_from_roots(state);
    state->needs_redraw = true;

  } else if (state->drag_mode == DRAG_COEFF && state->drag_index >= 0) {
    float cx, cy;
    screen_to_complex(state, mx, my, &cx, &cy);

    if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
      state->base_coeffs[state->drag_index] = CXL(cx, cy);
      // during dragging, skip combinations if there are too many
      size_t skip = 1;
      if (state->num_combinations > DRAG_RENDER_POINTS) {
        skip = (state->num_combinations + DRAG_RENDER_POINTS - 1) / DRAG_RENDER_POINTS;
      }
      state->drag_skip = skip;
      // use incremental solver during drag
      cloud_update_incremental(state, skip);
    } else {
      state->poly->coeffs[state->drag_index] = CXL(cx, cy);
      app_rebuild_from_coeffs(state);
    }
    state->needs_redraw = true;

  } else {
    // update hover state
    DragMode old_mode = state->hover_mode;
    int old_index = state->hover_index;
    input_find_hover(state, mx, my);
    if (state->hover_mode != old_mode || state->hover_index != old_index) {
      state->needs_redraw = true;
    }
  }
}

// handle mouse wheel
static void handle_mouse_wheel(AppState *state, SDL_Event *event) {
  float mx = event->wheel.mouse_x;
  float my = event->wheel.mouse_y;

  float complex_x, complex_y;
  screen_to_complex(state, mx, my, &complex_x, &complex_y);

  float zoom_factor = (event->wheel.y > 0) ? 0.9f : 1.1f;
  state->scale *= zoom_factor;

  if (state->scale < 1e-6f) state->scale = 1e-6f;
  if (state->scale > 1.0f) state->scale = 1.0f;

  state->center_x = complex_x - (mx - (float)state->width * 0.5f) * state->scale;
  state->center_y = complex_y - (my - (float)state->height * 0.5f) * state->scale;

  state->needs_redraw = true;
}

// handle window resize
static void handle_window_resize(AppState *state) {
  int new_width, new_height;
  SDL_GetRenderOutputSize(state->ren, &new_width, &new_height);

  if (new_width != state->width || new_height != state->height) {
    SDL_LogDebug(SDL_LOG_CATEGORY_VIDEO, "window resized: %dx%d", new_width, new_height);

    state->width = new_width;
    state->height = new_height;

    resize_buffers(state->width, state->height);

    if (state->texture) {
      SDL_DestroyTexture(state->texture);
    }
    state->texture = SDL_CreateTexture(state->ren, SDL_PIXELFORMAT_RGBA32,
                                       SDL_TEXTUREACCESS_STREAMING,
                                       state->width, state->height);
    state->needs_redraw = true;
  }
}

SDL_AppResult input_handle_event(AppState *state, SDL_Event *event) {
  switch (event->type) {
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
      handle_window_resize(state);
      break;

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
      handle_mouse_down(state, event);
      break;

    case SDL_EVENT_MOUSE_BUTTON_UP:
      handle_mouse_up(state, event);
      break;

    case SDL_EVENT_MOUSE_MOTION:
      handle_mouse_motion(state, event);
      break;

    case SDL_EVENT_MOUSE_WHEEL:
      handle_mouse_wheel(state, event);
      break;

    case SDL_EVENT_KEY_UP:
      handle_key_up(state, event);
      break;

    case SDL_EVENT_KEY_DOWN:
      return handle_key_down(state, event);

    case SDL_EVENT_QUIT:
      return SDL_APP_SUCCESS;

    default:
      break;
  }

  return SDL_APP_CONTINUE;
}

void input_process_continuous(AppState *state) {
  bool panning = false;
  bool zooming = false;

  for (int i = 0; i < 4; i++) {
    panning |= state->pan_keys[i];
  }
  for (int i = 0; i < 2; i++) {
    zooming |= state->zoom_keys[i];
  }

  if (!panning) {
    state->pan_accel = 1.0f;
    state->pan_repeat = 0;
  } else {
    if (state->pan_keys[0]) {
      state->center_y -= PAN_BASE_RATE * state->pan_accel * state->scale;
    }
    if (state->pan_keys[1]) {
      state->center_x -= PAN_BASE_RATE * state->pan_accel * state->scale;
    }
    if (state->pan_keys[2]) {
      state->center_y += PAN_BASE_RATE * state->pan_accel * state->scale;
    }
    if (state->pan_keys[3]) {
      state->center_x += PAN_BASE_RATE * state->pan_accel * state->scale;
    }
    state->pan_repeat++;
    if (state->pan_repeat > PAN_REPEAT_THRESH && state->pan_accel < PAN_ACCEL_MAX) {
      state->pan_accel *= PAN_ACCEL;
    }
    state->needs_redraw = true;
  }

  if (!zooming) {
    state->zoom_accel = 1.0f;
    state->zoom_repeat = 0;
  } else {
    if (state->zoom_keys[0]) {
      state->scale += ZOOM_BASE_RATE * state->scale * state->zoom_accel;
      if (state->zoom_dir) state->zoom_repeat = 0;
    }
    if (state->zoom_keys[1]) {
      state->scale -= ZOOM_BASE_RATE * state->scale * state->zoom_accel;
      if (!state->zoom_dir) state->zoom_repeat = 0;
    }

    state->zoom_repeat++;
    if (state->zoom_repeat > ZOOM_REPEAT_THRESH && state->zoom_accel < ZOOM_ACCEL_MAX) {
      state->zoom_accel *= ZOOM_ACCEL;
    }
    if (state->scale < 1e-6f) state->scale = 1e-6f;
    if (state->scale > 1.0f) state->scale = 1.0f;
    state->needs_redraw = true;
  }
}