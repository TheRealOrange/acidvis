//
// Created by Lin Yicheng on 30/11/25.
//

#include "cloud.h"
#include "combination.h"
#include "util.h"

#include <SDL3/SDL.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void cloud_reallocate_combinations(AppState *state) {
  if (state->combination_roots) {
    free(state->combination_roots);
    state->combination_roots = NULL;
  }
  if (state->combination_valid) {
    free(state->combination_valid);
    state->combination_valid = NULL;
  }
  if (state->prev_base_coeffs) {
    free(state->prev_base_coeffs);
    state->prev_base_coeffs = NULL;
  }
  if (state->since_last_update) {
    free(state->since_last_update);
    state->since_last_update = NULL;
  }

  size_t num_combs = int_pow(state->num_base_coeffs, state->poly_degree_cloud + 1);
  SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION,
               "allocating deg=%zu combinations=%zu", state->poly_degree_cloud, num_combs);

  state->combination_roots = malloc(num_combs * state->poly_degree_cloud * sizeof(cxldouble));
  if (!state->combination_roots) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate roots array");
    return;
  }

  state->combination_valid = malloc(num_combs * sizeof(bool));
  if (!state->combination_valid) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate combination_valid array");
    free(state->combination_roots);
    state->combination_roots = NULL;
    return;
  }

  // allocate caching arrays for incremental solving
  state->prev_base_coeffs = malloc(state->num_base_coeffs * sizeof(cxldouble));
  if (!state->prev_base_coeffs) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate prev_base_coeffs array");
    free(state->combination_roots);
    free(state->combination_valid);
    state->combination_roots = NULL;
    state->combination_valid = NULL;
    return;
  }

  state->since_last_update = malloc(num_combs * sizeof(int));
  if (!state->since_last_update) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate since_last_update array");
    free(state->combination_roots);
    free(state->combination_valid);
    free(state->prev_base_coeffs);
    state->combination_roots = NULL;
    state->combination_valid = NULL;
    state->prev_base_coeffs = NULL;
    return;
  }

  // initialize since_last_update to trigger full solve on first update
  for (size_t i = 0; i < num_combs; i++) {
    state->since_last_update[i] = -1;
  }

  state->num_combinations = num_combs;
  state->comb_roots_stride = state->poly_degree_cloud;
}

// full solve, no caching
void cloud_update(AppState *state, size_t skip) {
  if (!state->base_coeffs || state->view_mode != VIEW_MODE_POINT_CLOUD) return;
  if (state->num_base_coeffs == 0 || state->poly_degree_cloud == 0) {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                "cloud update failed: num_coeffs=%zu deg=%zu",
                state->num_base_coeffs, state->poly_degree_cloud);
    return;
  }

  bool use_lapack = false;
#ifdef HAVE_LAPACK
  use_lapack = state->lapack;
#endif

  state->found_roots = polynomial_find_root_combinations(
    state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
    state->combination_roots, state->combination_valid,
    state->since_last_update, state->num_combinations,
    skip, use_lapack
  );

  // save current coefficients for incremental solving
  if (state->prev_base_coeffs) {
    memcpy(state->prev_base_coeffs, state->base_coeffs,
           state->num_base_coeffs * sizeof(cxldouble));
  }
}

// incremental solve during drag
void cloud_update_incremental(AppState *state, size_t skip) {
  if (!state->base_coeffs || state->view_mode != VIEW_MODE_POINT_CLOUD) return;
  if (state->num_base_coeffs == 0 || state->poly_degree_cloud == 0) {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                "cloud update failed: num_coeffs=%zu deg=%zu",
                state->num_base_coeffs, state->poly_degree_cloud);
    return;
  }

  // if we don't have previous coefficients yet, fall back to full solve
  if (!state->prev_base_coeffs || !state->since_last_update) {
    cloud_update(state, skip);
    return;
  }

  bool use_lapack = false;
#ifdef HAVE_LAPACK
  use_lapack = state->lapack;
#endif

  state->found_roots = polynomial_find_root_combinations_cached(
    state->base_coeffs, state->prev_base_coeffs,
    state->num_base_coeffs, state->poly_degree_cloud,
    state->combination_roots, state->combination_valid,
    state->since_last_update, state->num_combinations,
    skip, use_lapack
  );

  // save current coefficients for next incremental solve
  memcpy(state->prev_base_coeffs, state->base_coeffs,
         state->num_base_coeffs * sizeof(cxldouble));
}

// call after drag ends for full recalculation
void cloud_update_drag_end(AppState *state) {
  cloud_update(state, 1);
}

void cloud_mode_enter(AppState *state) {
  state->view_mode = VIEW_MODE_POINT_CLOUD;

  if (!state->base_coeffs || int_pow(state->num_base_coeffs, state->poly_degree_cloud + 1) > 1000000) {
    // reset to safe defaults
    state->num_base_coeffs = 2;
    state->base_coeffs = malloc(state->num_base_coeffs * sizeof(cxldouble));
    for (size_t i = 0; i < state->num_base_coeffs; i++) {
      long double angle = ((long double)i * M_PI * 2.0L) / (long double)state->num_base_coeffs;
      state->base_coeffs[i] = cxexpl(cxscalel(CXL_I, angle));
    }
    state->poly_degree_cloud = 2;
  }

  cloud_reallocate_combinations(state);
  cloud_update(state, 1);

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
              "entered cloud mode: %zu coeffs, degree %zu, %zu combinations",
              state->num_base_coeffs, state->poly_degree_cloud, state->num_combinations);
}

void cloud_mode_exit(AppState *state) {
  if (state->combination_roots) {
    free(state->combination_roots);
    state->combination_roots = NULL;
  }
  if (state->combination_valid) {
    free(state->combination_valid);
    state->combination_valid = NULL;
  }
  if (state->prev_base_coeffs) {
    free(state->prev_base_coeffs);
    state->prev_base_coeffs = NULL;
  }
  if (state->since_last_update) {
    free(state->since_last_update);
    state->since_last_update = NULL;
  }
  state->num_combinations = 0;

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "exited cloud mode");
}