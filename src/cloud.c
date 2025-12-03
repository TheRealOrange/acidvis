//
// Created by Lin Yicheng on 30/11/25.
//

#include "cloud.h"
#include "polynomial.h"
#include "util.h"

#include <SDL3/SDL.h>
#include <math.h>
#include <stdlib.h>

void cloud_reallocate_combinations(AppState *state) {
  if (state->combination_roots) {
    free(state->combination_roots);
    state->combination_roots = NULL;
  }
  if (state->num_distinct) {
    free(state->num_distinct);
    state->num_distinct = NULL;
  }

  size_t num_combs = int_pow(state->num_base_coeffs, state->poly_degree_cloud + 1);
  SDL_LogDebug(SDL_LOG_CATEGORY_APPLICATION,
               "allocating deg=%zu combinations=%zu", state->poly_degree_cloud, num_combs);

  state->combination_roots = malloc(num_combs * state->poly_degree_cloud * sizeof(cxldouble));
  if (!state->combination_roots) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate roots array");
    return;
  }

  state->num_distinct = malloc(num_combs * sizeof(size_t));
  if (!state->num_distinct) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate num_distinct array");
    free(state->combination_roots);
    state->combination_roots = NULL;
    return;
  }

  state->num_combinations = num_combs;
  state->comb_roots_stride = state->poly_degree_cloud;
}

void cloud_update(AppState *state, size_t skip) {
  if (!state->base_coeffs || state->view_mode != VIEW_MODE_POINT_CLOUD) return;
  if (state->num_base_coeffs == 0 || state->poly_degree_cloud == 0) {
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                "cloud update failed: num_coeffs=%zu deg=%zu",
                state->num_base_coeffs, state->poly_degree_cloud);
    return;
  }

#ifdef HAVE_LAPACK
  if (state->lapack) {
    if (skip > 1) {
      state->found_roots = polynomial_find_root_combinations_companion_skip(
          state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
          &state->combination_roots, &state->num_distinct, state->num_combinations, skip);
    } else {
      state->found_roots = polynomial_find_root_combinations_companion(
          state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
          &state->combination_roots, &state->num_distinct, state->num_combinations);
    }
  } else {
    if (skip > 1) {
      state->found_roots = polynomial_find_root_combinations_skip(
          state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
          &state->combination_roots, &state->num_distinct, state->num_combinations, skip);
    } else {
      state->found_roots = polynomial_find_root_combinations(
          state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
          &state->combination_roots, &state->num_distinct, state->num_combinations);
    }
  }
#else
  if (skip > 1) {
    state->found_roots = polynomial_find_root_combinations_skip(
        state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
        &state->combination_roots, &state->num_distinct, state->num_combinations, skip);
  } else {
    state->found_roots = polynomial_find_root_combinations(
        state->base_coeffs, state->num_base_coeffs, state->poly_degree_cloud,
        &state->combination_roots, &state->num_distinct, state->num_combinations);
  }
#endif
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
  if (state->num_distinct) {
    free(state->num_distinct);
    state->num_distinct = NULL;
  }
  state->num_combinations = 0;

  SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "exited cloud mode");
}
