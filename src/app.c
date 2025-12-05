//
// Created by Lin Yicheng on 30/11/25.
//

#include "app.h"
#include "cloud.h"
#include "lapacksolve.h"
#include "render.h"
#include "util.h"

#include <SDL3/SDL.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// create initial polynomial with roots on unit circle
polynomial_t *app_create_default_polynomial(size_t degree) {
  cxldouble *roots = calloc(degree, sizeof(cxldouble));
  if (!roots) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate roots array");
    return NULL;
  }

  // initialize roots on unit circle
  for (size_t i = 0; i < degree; i++) {
    long double angle = ((long double)i * M_PI * 2.0L) / (long double)degree;
    roots[i] = cxexpl(cxscalel(CXL_I, angle));
  }

  polynomial_t *p = polynomial_from_roots(roots, degree, false);
  if (p) {
    polynomial_find_roots(p, true);  // ensure roots are valid
  }

  free(roots);
  return p;
}

// rebuild polynomial from modified roots
void app_rebuild_from_roots(AppState *state) {
  if (!state->poly || !state->poly->roots_valid) return;

  // expand roots with multiplicities
  size_t total_roots = 0;
  for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
    total_roots += state->poly->multiplicity[i];
  }

  cxldouble *all_roots = malloc(total_roots * sizeof(cxldouble));
  if (!all_roots) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate roots for rebuild");
    return;
  }

  size_t idx = 0;
  for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
    for (size_t m = 0; m < state->poly->multiplicity[i]; m++) {
      all_roots[idx++] = state->poly->roots[i];
    }
  }

  // create new polynomial from roots
  polynomial_t *new_poly = polynomial_from_roots(all_roots, total_roots, false);
  free(all_roots);

  if (new_poly) {
    polynomial_free(state->poly);
    state->poly = new_poly;
    set_polynomial(state->poly);
  }
}

// rebuild polynomial from modified coefficients
void app_rebuild_from_coeffs(AppState *state) {
  if (!state->poly || !state->poly->coeffs_valid) return;

  // find roots of the modified polynomial
#ifdef HAVE_LAPACK
  if (state->lapack)
    polynomial_find_roots_companion(state->poly, true);
  else
    polynomial_find_roots(state->poly, true);
#else
  polynomial_find_roots(state->poly, true);
#endif

  set_polynomial(state->poly);
}

bool app_editing_allowed(AppState *state) {
  // editing is blocked when animation is active
  return !state->anim_active;
}

void app_update_animation(AppState *state) {
  if (!state->anim_active || !state->anim_state) return;
  if (!state->anim_state->playing) return;

  // calculate delta time
  Uint64 now = SDL_GetTicks();
  float delta = (float)(now - state->anim_last_tick) / 1000.0f;
  state->anim_last_tick = now;

  // clamp to avoid jumps after pause/resume or window focus loss
  if (delta > 0.1f) delta = 0.1f;

  if (!anim_update(state->anim_state, delta)) {
    return;
  }

  // apply interpolated points
  anim_state_t *anim = state->anim_state;

  if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
    // update base coefficients from animation
    for (size_t i = 0; i < anim->num_points && i < state->num_base_coeffs; i++) {
      state->base_coeffs[i] = anim->points[i];
    }
    cloud_update(state, 1);
  } else {
    // update roots directly
    if (state->poly && state->poly->roots_valid) {
      for (size_t i = 0; i < anim->num_points && i < state->poly->num_distinct_roots; i++) {
        state->poly->roots[i] = anim->points[i];
      }
      app_rebuild_from_roots(state);
    }
  }

  state->needs_redraw = true;
}

bool app_load_animation(AppState *state, const char *filename) {
  // free existing
  if (state->anim_state) {
    anim_free_state(state->anim_state);
    state->anim_state = NULL;
  }
  if (state->anim_script) {
    anim_free_script(state->anim_script);
    state->anim_script = NULL;
  }
  state->anim_active = false;

  // load script
  state->anim_script = anim_load_script(filename);
  if (!state->anim_script) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to load animation: %s", filename);
    return false;
  }

  // create playback state
  state->anim_state = anim_create_state(state->anim_script);
  if (!state->anim_state) {
    anim_free_script(state->anim_script);
    state->anim_script = NULL;
    return false;
  }

  // configure mode and allocate resources
  if (state->anim_script->mode == ANIM_MODE_CLOUD) {
    state->view_mode = VIEW_MODE_POINT_CLOUD;

    // set degree from script
    state->poly_degree_cloud = state->anim_script->degree;

    // resize base_coeffs if needed
    size_t needed = state->anim_script->num_coeffs;
    if (needed > state->num_base_coeffs) {
      free(state->base_coeffs);
      state->base_coeffs = malloc(needed * sizeof(cxldouble));
      state->num_base_coeffs = needed;
    } else if (needed > 0 && needed < state->num_base_coeffs) {
      state->num_base_coeffs = needed;
    }

    cloud_reallocate_combinations(state);

    // apply initial points
    for (size_t i = 0; i < state->anim_state->num_points && i < state->num_base_coeffs; i++) {
      state->base_coeffs[i] = state->anim_state->points[i];
    }
    cloud_update(state, 1);
  } else {
    // switch to roots mode if needed
    if (state->view_mode != VIEW_MODE_ROOTS) {
      state->view_mode = VIEW_MODE_ROOTS;
      cloud_mode_exit(state);
    }

    // apply initial points to polynomial roots
    if (state->poly && state->poly->roots_valid) {
      for (size_t i = 0; i < state->anim_state->num_points && i < state->poly->num_distinct_roots; i++) {
        state->poly->roots[i] = state->anim_state->points[i];
      }
      app_rebuild_from_roots(state);
    }
  }

  strncpy(state->anim_filename, filename, sizeof(state->anim_filename) - 1);
  state->anim_active = true;
  state->anim_last_tick = SDL_GetTicks();
  state->needs_redraw = true;

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
              "loaded animation: %s (%.1fs, %zu keyframes, %s)",
              filename, state->anim_script->total_duration,
              state->anim_script->num_keyframes,
              state->anim_script->loop ? "looping" : "once");

  return true;
}

void app_unload_animation(AppState *state) {
  if (state->anim_state) {
    anim_free_state(state->anim_state);
    state->anim_state = NULL;
  }
  if (state->anim_script) {
    anim_free_script(state->anim_script);
    state->anim_script = NULL;
  }
  state->anim_active = false;
  state->anim_filename[0] = '\0';

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "animation unloaded");
}

bool app_init(AppState *state, int argc, char **argv) {
  memset(state, 0, sizeof(AppState));

  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to init sdl: %s", SDL_GetError());
    return false;
  }

  state->win = SDL_CreateWindow("polynomial visualizer", 800, 600, SDL_WINDOW_RESIZABLE);
  if (!state->win) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to create window: %s", SDL_GetError());
    return false;
  }

  state->ren = SDL_CreateRenderer(state->win, NULL);
  if (!state->ren) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to create renderer: %s", SDL_GetError());
    return false;
  }

  SDL_SetRenderVSync(state->ren, 1);

  if (!initialize_renderer()) {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "gpu renderer unavailable, using cpu fallback");
  }

#ifdef HAVE_LAPACK
  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "lapack solver available");
#endif
  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "using render device: %s", get_device_name());
#ifdef _OPENMP
  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "openmp threads: %d", get_num_threads());
#endif

  state->poly = app_create_default_polynomial(4);
  if (!state->poly) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to create polynomial");
    return false;
  }

  set_polynomial(state->poly);

  // view state defaults
  state->scale = 0.005f;
  state->center_x = 0.0f;
  state->center_y = 0.0f;
  state->view_mode = VIEW_MODE_ROOTS;
  state->drag_mode = DRAG_NONE;
  state->hover_mode = DRAG_NONE;
  state->hover_index = -1;
  state->needs_redraw = true;

  state->show_roots = true;
  state->show_coeffs = true;
  state->lapack = true;
  state->show_info = true;
  state->fixed_point_scale = false;
  state->cloud_point_radius = 1.0f;
  state->drag_skip = 1;

  // point cloud defaults
  state->num_base_coeffs = 3;
  state->poly_degree_cloud = 4;
  state->base_coeffs = malloc(state->num_base_coeffs * sizeof(cxldouble));
  if (state->base_coeffs) {
    for (size_t i = 0; i < state->num_base_coeffs; i++) {
      long double angle = ((long double)i * M_PI * 2.0L) / (long double)state->num_base_coeffs;
      state->base_coeffs[i] = cxexpl(cxscalel(CXL_I, angle));
    }
  }

  // caching arrays initialized to NULL, will be allocated when entering cloud mode
  state->prev_base_coeffs = NULL;
  state->since_last_update = NULL;

  // keyboard state
  state->zoom_accel = 1.0f;
  state->pan_accel = 1.0f;

  // get window size and create texture
  SDL_GetRenderOutputSize(state->ren, &state->width, &state->height);
  resize_buffers(state->width, state->height);

  state->texture = SDL_CreateTexture(state->ren, SDL_PIXELFORMAT_RGBA32,
                                     SDL_TEXTUREACCESS_STREAMING,
                                     state->width, state->height);
  if (!state->texture) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to create texture: %s", SDL_GetError());
    return false;
  }

  // load animation from command line if provided
  if (argc > 1) {
    const char *ext = strrchr(argv[1], '.');
    if (ext && strcmp(ext, ".json") == 0) {
      app_load_animation(state, argv[1]);
    }
  }

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "initialization complete");
  return true;
}

void app_cleanup(AppState *state) {
  if (!state) return;

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "cleaning up");

  if (state->anim_state) anim_free_state(state->anim_state);
  if (state->anim_script) anim_free_script(state->anim_script);
  if (state->texture) SDL_DestroyTexture(state->texture);
  if (state->poly) polynomial_free(state->poly);
  if (state->combination_roots) free(state->combination_roots);
  if (state->combination_valid) free(state->combination_valid);
  if (state->base_coeffs) free(state->base_coeffs);
  if (state->prev_base_coeffs) free(state->prev_base_coeffs);
  if (state->since_last_update) free(state->since_last_update);

  cleanup_renderer();
}