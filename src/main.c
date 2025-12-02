#include "polynomial.h"
#include "render.h"

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <SDL3/SDL.h>

#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>

#include "animation.h"
#include "companion.h"
#include "util.h"

#define SCANCODE_RESET_VIEW SDL_SCANCODE_R
#define SCANCODE_RESET_POLY SDL_SCANCODE_C

#define SCANCODE_INC_DEGREE SDL_SCANCODE_Y
#define SCANCODE_DEC_DEGREE SDL_SCANCODE_H

#define SCANCODE_ADD_COEFF   SDL_SCANCODE_EQUALS
#define SCANCODE_REM_COEFF   SDL_SCANCODE_MINUS

#define SCANCODE_PAN_UP      SDL_SCANCODE_I
#define SCANCODE_PAN_LEFT    SDL_SCANCODE_J
#define SCANCODE_PAN_DOWN    SDL_SCANCODE_K
#define SCANCODE_PAN_RIGHT   SDL_SCANCODE_L

#define SCANCODE_INC_ZOOM    SDL_SCANCODE_U
#define SCANCODE_DEC_ZOOM    SDL_SCANCODE_O

#define SCANCODE_INC_GAMMA   SDL_SCANCODE_N
#define SCANCODE_DEC_GAMMA   SDL_SCANCODE_M

#define SCANCODE_TOGGLE_MODE SDL_SCANCODE_T
#define SCANCODE_INVERT      SDL_SCANCODE_V
#define SCANCODE_LOG_SCALE   SDL_SCANCODE_Z

#define SCANCODE_TOGGLE_ROOTS  SDL_SCANCODE_Q
#define SCANCODE_TOGGLE_COEFFS SDL_SCANCODE_W

#define SCANCODE_LAPACK      SDL_SCANCODE_B
#define SCANCODE_TOGGLE_INFO SDL_SCANCODE_F1
#define SCANCODE_SCALE_MODE  SDL_SCANCODE_S

#define SCANCODE_ANIM_PLAY   SDL_SCANCODE_SPACE
#define SCANCODE_ANIM_RESET  SDL_SCANCODE_X

#define SCANCODE_QUIT        SDL_SCANCODE_ESCAPE

#define DRAG_RENDER_POINTS 20000

#define POINT_RADIUS       10.0f
#define POINT_HIT_RADIUS   12.0f
#define CLOUD_POINT_RADIUS_DEFAULT 1.0f
#define CLOUD_POINT_RADIUS_MIN     0.2f
#define CLOUD_POINT_RADIUS_MAX     10.0f

#define MAX_DEGREE 26

#define ZOOM_ACCEL         1.08f
#define ZOOM_BASE_RATE     0.03f
#define ZOOM_ACCEL_MAX     10.0f
#define ZOOM_REPEAT_THRESH 5

#define PAN_ACCEL         1.08f
#define PAN_BASE_RATE     5.0f
#define PAN_ACCEL_MAX     5.0f
#define PAN_REPEAT_THRESH 5

typedef enum {
  DRAG_NONE,
  DRAG_PAN,
  DRAG_ROOT,
  DRAG_COEFF
} DragMode;

typedef enum {
  VIEW_MODE_ROOTS,
  VIEW_MODE_POINT_CLOUD
} ViewMode;

typedef struct {
  SDL_Window* win;
  SDL_Renderer* ren;
  SDL_Texture* texture;
  polynomial_t* poly;

  // view parameters
  float scale;     // units per pixel in complex plane
  float center_x;  // center of view in complex plane
  float center_y;

  // view mode
  ViewMode view_mode;

  bool lapack;
  bool show_info;
  bool fixed_point_scale;  // true = fixed radius, false = scales with zoom
  float cloud_point_radius;
  size_t drag_skip;  // current skip factor during drag
  bool show_roots;   // show root point markers
  bool show_coeffs;  // show coefficient point markers

  // base coefficients for VIEW_MODE_POINT_CLOUD (the N draggable points)
  complex long double* base_coeffs;
  size_t num_base_coeffs;
  size_t poly_degree_cloud;

  // combination cloud roots
  complex long double* combination_roots;
  size_t *num_distinct;
  size_t comb_roots_stride;
  size_t num_combinations;  // N^(M+1)
  size_t found_roots;

  // drag state
  DragMode drag_mode;
  int drag_index;  // which root or coeff is being dragged
  float drag_start_mouse_x;
  float drag_start_mouse_y;
  float drag_start_center_x;
  float drag_start_center_y;

  // keyboard zooom
  float zoom_accel;
  int zoom_repeat;
  bool zoom_dir;
  bool zoom_keys[2];

  // keyboard pan
  float pan_accel;
  int pan_repeat;
  bool pan_keys[4];

  // hover state for visual feedback
  DragMode hover_mode;
  int hover_index;

  bool needs_redraw;
  int width;
  int height;

  // animation playback
  anim_script_t *anim_script;
  anim_state_t *anim_state;
  bool anim_active;              // true when animation is loaded
  char anim_filename[512];       // for reload functionality
  Uint64 anim_last_tick;         // delta time tracking
} AppState;

// coordinate conversion helpers

static void screen_to_complex(AppState* state, float sx, float sy, float* cx, float* cy) {
  *cx = (sx - state->width * 0.5f) * state->scale + state->center_x;
  *cy = (sy - state->height * 0.5f) * state->scale + state->center_y;
}

static void complex_to_screen(AppState* state, float cx, float cy, float* sx, float* sy) {
  *sx = (cx - state->center_x) / state->scale + state->width * 0.5f;
  *sy = (cy - state->center_y) / state->scale + state->height * 0.5f;
}

// draw a filled circle
static void draw_circle(SDL_Renderer* ren, float cx, float cy, float r) {
  // simple scanline fill
  int ir = (int)ceilf(r);
  for (int dy = -ir; dy <= ir; dy++) {
    float dx = sqrtf(r * r - dy * dy);
    SDL_RenderLine(ren, cx - dx, cy + dy, cx + dx, cy + dy);
  }
}

// draw a circle outline
static void draw_circle_outline(SDL_Renderer* ren, float cx, float cy, float r, int segments) {
  float prev_x = cx + r;
  float prev_y = cy;
  for (int i = 1; i <= segments; i++) {
    float angle = (float)i / segments * 2.0f * (float)M_PI;
    float x = cx + r * cosf(angle);
    float y = cy + r * sinf(angle);
    SDL_RenderLine(ren, prev_x, prev_y, x, y);
    prev_x = x;
    prev_y = y;
  }
}

// check if screen point is near a complex point
static bool point_hit_test(AppState* state, float mouse_x, float mouse_y,
                           float complex_x, float complex_y) {
  float sx, sy;
  complex_to_screen(state, complex_x, complex_y, &sx, &sy);
  float dx = mouse_x - sx;
  float dy = mouse_y - sy;
  return (dx * dx + dy * dy) <= POINT_HIT_RADIUS * POINT_HIT_RADIUS;
}

// find what point (if any) is under the mouse
static void find_hover(AppState* state, float mouse_x, float mouse_y) {
  state->hover_mode = DRAG_NONE;
  state->hover_index = -1;

  if (!state->poly) return;

  // check roots (only in roots mode)
  if (state->view_mode == VIEW_MODE_ROOTS && state->poly->roots_valid) {
    for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
      float rx = (float)creall(state->poly->roots[i]);
      float ry = (float)cimagl(state->poly->roots[i]);
      if (point_hit_test(state, mouse_x, mouse_y, rx, ry)) {
        state->hover_mode = DRAG_ROOT;
        state->hover_index = (int)i;
        return;
      }
    }
  }

  // in cloud mode check base_coeffs
  if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->base_coeffs && state->show_coeffs) {
    for (size_t i = 0; i < state->num_base_coeffs; i++) {
      float cx = (float)creall(state->base_coeffs[i]);
      float cy = (float)cimagl(state->base_coeffs[i]);
      if (point_hit_test(state, mouse_x, mouse_y, cx, cy)) {
        state->hover_mode = DRAG_COEFF;
        state->hover_index = (int)i;
        return;
      }
    }
  } else if (state->poly->coeffs_valid && state->show_coeffs) {
    for (size_t i = 0; i <= state->poly->degree; i++) {
      float cx = (float)creall(state->poly->coeffs[i]);
      float cy = (float)cimagl(state->poly->coeffs[i]);
      if (point_hit_test(state, mouse_x, mouse_y, cx, cy)) {
        state->hover_mode = DRAG_COEFF;
        state->hover_index = (int)i;
        return;
      }
    }
  }
}



// rebuild polynomial from modified roots
static void rebuild_from_roots(AppState* state) {
  if (!state->poly || !state->poly->roots_valid) return;

  // expand roots with multiplicities
  size_t total_roots = 0;
  for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
    total_roots += state->poly->multiplicity[i];
  }

  complex long double* all_roots = malloc(total_roots * sizeof(complex long double));
  size_t idx = 0;
  for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
    for (size_t m = 0; m < state->poly->multiplicity[i]; m++) {
      all_roots[idx++] = state->poly->roots[i];
    }
  }

  // create new polynomial from roots
  polynomial_t* new_poly = polynomial_from_roots(all_roots, total_roots, false);
  free(all_roots);

  if (new_poly) {
    polynomial_free(state->poly);
    state->poly = new_poly;
    set_polynomial(state->poly);
  }
}

// rebuild polynomial from modified coefficients
static void rebuild_from_coeffs(AppState* state) {
  if (!state->poly || !state->poly->coeffs_valid) return;

  // find roots of the modified polynomial
#ifdef HAVE_LAPACK
  if (state->lapack)
    polynomial_find_roots_companion(state->poly);
  else
    polynomial_find_roots(state->poly);
#else
  polynomial_find_roots(state->poly);
#endif
  set_polynomial(state->poly);
}

static void reallocate_combinations(AppState* state) {
  if (state->combination_roots) {
    free(state->combination_roots);
    state->combination_roots = nullptr;
  }
  if (state->num_distinct) {
    free(state->num_distinct);
    state->num_distinct = nullptr;
  }

  size_t num_combs = int_pow(state->num_base_coeffs, state->poly_degree_cloud + 1);
  printf("allocating deg=%lu combinations=%lu\n", state->poly_degree_cloud, num_combs);
  state->combination_roots = malloc(num_combs * state->poly_degree_cloud * sizeof(complex long double));
  if (!state->combination_roots) {
    SDL_Log("failed to allocate roots array");
    return;
  }

  state->num_distinct = malloc(num_combs * sizeof(size_t));
  if (!state->num_distinct) {
    SDL_Log("failed to allocate num_distinct array");
    free(state->combination_roots);
    state->combination_roots = nullptr;
    return;
  }

  state->num_combinations = num_combs;
  state->comb_roots_stride = state->poly_degree_cloud;
}

// update point cloud when coefficients change
// skip parameter: if > 1, only computes every skip-th combination for faster dragging
static void update_combination_cloud(AppState* state, size_t skip) {
  if (!state->base_coeffs || state->view_mode != VIEW_MODE_POINT_CLOUD) return;
  if (state->num_base_coeffs == 0 || state->poly_degree_cloud == 0) {
    printf("update combination cloud failed num coeff %lu deg %lu\n",
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

// create initial polynomial
static polynomial_t* create_default_polynomial(size_t degree) {
  complex long double *roots = calloc(degree, sizeof(complex long double));
  // initialize roots on unit circle
  for (size_t i = 0; i < degree; i++) {
    long double angle = ((long double)i * M_PI * 2.0l)/(long double)degree;
    roots[i] = cexpl(I * angle);
  }
  polynomial_t* p = polynomial_from_roots(roots, degree, false);
  if (p) {
    polynomial_find_roots(p);  // ensure roots are valid
  }
  return p;
}

static bool editing_allowed(AppState *state) {
  // editing is blocked when animation is active
  return !state->anim_active;
}

static void update_animation(AppState *state) {
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
    update_combination_cloud(state, 1);
  } else {
    // update roots directly
    if (state->poly && state->poly->roots_valid) {
      for (size_t i = 0; i < anim->num_points && i < state->poly->num_distinct_roots; i++) {
        state->poly->roots[i] = anim->points[i];
      }
      rebuild_from_roots(state);
    }
  }

  // if (anim->view_changed) {
  //   state->scale = anim->view.scale;
  //   state->center_x = anim->view.center_x;
  //   state->center_y = anim->view.center_y;
  // }

  state->needs_redraw = true;
}


static bool load_animation_file(AppState *state, const char *filename) {
  // free existing
  if (state->anim_state) {
    anim_free_state(state->anim_state);
    state->anim_state = nullptr;
  }
  if (state->anim_script) {
    anim_free_script(state->anim_script);
    state->anim_script = nullptr;
  }
  state->anim_active = false;

  // load script
  state->anim_script = anim_load_script(filename);
  if (!state->anim_script) {
    SDL_Log("failed to load animation: %s", filename);
    return false;
  }

  // create playback state
  state->anim_state = anim_create_state(state->anim_script);
  if (!state->anim_state) {
    anim_free_script(state->anim_script);
    state->anim_script = nullptr;
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
      state->base_coeffs = malloc(needed * sizeof(complex long double));
      state->num_base_coeffs = needed;
    } else if (needed > 0 && needed < state->num_base_coeffs) {
      state->num_base_coeffs = needed;
    }

    reallocate_combinations(state);

    // apply initial points
    for (size_t i = 0; i < state->anim_state->num_points && i < state->num_base_coeffs; i++) {
      state->base_coeffs[i] = state->anim_state->points[i];
    }
    update_combination_cloud(state, 1);
  } else {
    // switch to roots mode if needed
    if (state->view_mode != VIEW_MODE_ROOTS) {
      state->view_mode = VIEW_MODE_ROOTS;
      // free cloud resources
      if (state->combination_roots) {
        free(state->combination_roots);
        state->combination_roots = nullptr;
      }
      if (state->num_distinct) {
        free(state->num_distinct);
        state->num_distinct = nullptr;
      }
    }

    // apply initial points to polynomial roots
    if (state->poly && state->poly->roots_valid) {
      for (size_t i = 0; i < state->anim_state->num_points && i < state->poly->num_distinct_roots; i++) {
        state->poly->roots[i] = state->anim_state->points[i];
      }
      rebuild_from_roots(state);
    }
  }

  // uncomment to apply animation's starting view:
  // state->scale = state->anim_state->view.scale;
  // state->center_x = state->anim_state->view.center_x;
  // state->center_y = state->anim_state->view.center_y;

  strncpy(state->anim_filename, filename, sizeof(state->anim_filename) - 1);
  state->anim_active = true;
  state->anim_last_tick = SDL_GetTicks();
  state->needs_redraw = true;

  SDL_Log("loaded animation: %s (%.1fs, %zu keyframes, %s)",
          filename, state->anim_script->total_duration,
          state->anim_script->num_keyframes,
          state->anim_script->loop ? "looping" : "once");

  return true;
}

// unload animation and return to normal editing
static void unload_animation(AppState *state) {
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
  SDL_Log("animation unloaded");
}

SDL_AppResult SDL_AppInit(void** s, int argc, char** argv) {
  (void)argc;
  (void)argv;

  if (!((*s = SDL_calloc(1, sizeof(AppState))))) {
    return SDL_APP_FAILURE;
  }
  AppState* state = *s;

  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
    SDL_Log("failed to init sdl: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  if (!(state->win = SDL_CreateWindow("polynomial visualizer", 800, 600,
                                       SDL_WINDOW_RESIZABLE))) {
    SDL_Log("failed to create window: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  if (!(state->ren = SDL_CreateRenderer(state->win, NULL))) {
    SDL_Log("failed to create renderer: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  SDL_SetRenderVSync(state->ren, 1);

  if (!initialize_renderer()) {
    SDL_Log("failed to initialize polynomial renderer (using cpu fallback)");
  }

#ifdef HAVE_LAPACK
  SDL_Log("lapack solver available");
#endif
  SDL_Log("using render device: %s", get_device_name());
#ifdef _OPENMP
  SDL_Log("openmp threads: %d", get_num_threads());
#endif

  state->poly = create_default_polynomial(4);

  if (!state->poly) {
    SDL_Log("failed to create polynomial");
    return SDL_APP_FAILURE;
  }

  set_polynomial(state->poly);

  state->width = 0;
  state->height = 0;

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
  state->cloud_point_radius = CLOUD_POINT_RADIUS_DEFAULT;
  state->drag_skip = 1;

  state->comb_roots_stride = 0;
  state->combination_roots = nullptr;
  state->num_base_coeffs = 3;
  state->poly_degree_cloud = 4;
  state->base_coeffs = malloc(state->num_base_coeffs * sizeof(complex long double));
  if (state->base_coeffs) {
    for (size_t i = 0; i < state->num_base_coeffs; i++) {
      long double angle = ((long double)i * M_PI * 2.0L) / (long double)state->num_base_coeffs;
      state->base_coeffs[i] = cexpl(I * angle);
    }
  }

  state->zoom_accel = 1.0f;
  state->zoom_repeat = 0;
  state->zoom_dir = true;
  memset(state->zoom_keys, 0, sizeof(state->zoom_keys));

  state->pan_accel = 1.0f;
  state->pan_repeat = 0;
  memset(state->pan_keys, 0, sizeof(state->pan_keys));

  state->anim_script = nullptr;
  state->anim_state = nullptr;
  state->anim_active = false;
  state->anim_filename[0] = '\0';
  state->anim_last_tick = 0;

  SDL_GetRenderOutputSize(state->ren, &state->width, &state->height);
  resize_buffers(state->width, state->height);

  state->texture = SDL_CreateTexture(state->ren, SDL_PIXELFORMAT_RGBA32,
                                     SDL_TEXTUREACCESS_STREAMING,
                                     state->width, state->height);
  if (!state->texture) {
    SDL_Log("failed to create texture: %s", SDL_GetError());
    return SDL_APP_FAILURE;
  }

  // load animation from command line if provided
  if (argc > 1) {
    // check if it's a .json file
    const char *ext = strrchr(argv[1], '.');
    if (ext && strcmp(ext, ".json") == 0) {
      load_animation_file(state, argv[1]);
    }
  }

  return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* s) {
  AppState* state = s;

  // animation step, active if animation is active
  update_animation(state);

  if (state->needs_redraw) {
    // only render polynomial hue in roots mode
    if (state->view_mode == VIEW_MODE_ROOTS) {
      render_frame_roots(state->scale, state->center_x, state->center_y);
    } else {
      // clear to dark background
      clear_frame_buffer(20, 20, 25);
      // combination cloud mode, render point cloud using GPU/CPU
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

        render_point_cloud(state->combination_roots, state->num_distinct, state->num_combinations, state->comb_roots_stride, radius,
                           state->scale, state->center_x, state->center_y);
      }
    }

    const unsigned char* pixels = get_pixel_data();
    if (pixels) {
      SDL_UpdateTexture(state->texture, nullptr, pixels, state->width * 4);
    }

    SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
    SDL_RenderClear(state->ren);

    SDL_RenderTexture(state->ren, state->texture, nullptr, nullptr);

    float hue;
    uint8_t r, g, b;

    // draw coefficients or base_coeffs depending on mode
    if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->base_coeffs) {
      // cloud mode: draw base coefficients (the N draggable points)
      for (size_t i = 0; i < state->num_base_coeffs && i < 26; i++) {
        float cx = (float)creall(state->base_coeffs[i]);
        float cy = (float)cimagl(state->base_coeffs[i]);
        float sx, sy;
        complex_to_screen(state, cx, cy, &sx, &sy);

        bool is_hover = (state->hover_mode == DRAG_COEFF && state->hover_index == (int)i);
        bool is_drag = (state->drag_mode == DRAG_COEFF && state->drag_index == (int)i);

        hue = (float)i / (float)state->num_base_coeffs;
        if (is_hover || is_drag) {
          hsl_to_rgb_cpu(hue, 0.8f, 0.7f, &r, &g, &b);
          SDL_SetRenderDrawColor(state->ren, r, g, b, 255);
        } else {
          hsl_to_rgb_cpu(hue, 0.8f, 0.5f, &r, &g, &b);
          SDL_SetRenderDrawColor(state->ren, r, g, b, 255);
        }

        float rad = is_hover || is_drag ? POINT_RADIUS + 2 : POINT_RADIUS;
        draw_circle(state->ren, sx, sy, rad);

        // outline
        SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
        draw_circle_outline(state->ren, sx, sy, rad, 24);

        // draw label (a-z)
        char label[2] = { 'a' + (char)i, '\0' };
        SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
        SDL_RenderDebugText(state->ren, sx - 4, sy - 4, label);
      }
    } else if (state->poly && state->poly->coeffs_valid) {
      // roots mode: draw polynomial coefficients
      if (state->show_coeffs) {
        for (size_t i = 0; i <= state->poly->degree && i < 26; i++) {
          float cx = (float)creall(state->poly->coeffs[i]);
          float cy = (float)cimagl(state->poly->coeffs[i]);
          float sx, sy;
          complex_to_screen(state, cx, cy, &sx, &sy);

          bool is_hover = (state->hover_mode == DRAG_COEFF && state->hover_index == (int)i);
          bool is_drag = (state->drag_mode == DRAG_COEFF && state->drag_index == (int)i);

          hue = (float)i / (float)(state->poly->degree+1);
          if (is_hover || is_drag) {
            hsl_to_rgb_cpu(hue, 0.8f, 0.7f, &r, &g, &b);
            SDL_SetRenderDrawColor(state->ren, r, g, b, 255);
          } else {
            hsl_to_rgb_cpu(hue, 0.8f, 0.5f, &r, &g, &b);
            SDL_SetRenderDrawColor(state->ren, r, g, b, 255);
          }

          float rad = is_hover || is_drag ? POINT_RADIUS + 2 : POINT_RADIUS;
          draw_circle(state->ren, sx, sy, rad);

          // outline
          SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
          draw_circle_outline(state->ren, sx, sy, rad, 24);

          // draw label (a-z)
          char label[2] = { 'a' + (char)i, '\0' };
          SDL_SetRenderDrawColor(state->ren, 0, 0, 0, 255);
          SDL_RenderDebugText(state->ren, sx - 4, sy - 4, label);
        }
      }
    }

    // draw roots (only in roots mode)
    if (state->poly && state->poly->roots_valid) {
      if (state->view_mode == VIEW_MODE_ROOTS) {
        if (state->show_roots) {
          for (size_t i = 0; i < state->poly->num_distinct_roots; i++) {
            float rx = (float)creall(state->poly->roots[i]);
            float ry = (float)cimagl(state->poly->roots[i]);
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
            draw_circle(state->ren, sx, sy, rad);

            // outline
            SDL_SetRenderDrawColor(state->ren, 50, 50, 50, 255);
            draw_circle_outline(state->ren, sx, sy, rad, 24);
          }
        }
      }
    }

    // info overlay
    if (state->show_info) {
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

      if (state->anim_active && state->anim_state)  {

        const char *status = state->anim_state->playing ? "playing" :
                            (state->anim_state->finished ? "finished" : "paused");

        SDL_RenderDebugTextFormat(state->ren, 5, 70,
            "anim: %.1f/%.1fs %s%s",
            state->anim_state->current_time,
            state->anim_script->total_duration,
            status,
            state->anim_script->loop ? " [loop]" : "");
      }
    } else {
      // minimal hint when info is hidden
      SDL_SetRenderDrawColor(state->ren, 128, 128, 128, 255);
      SDL_RenderDebugTextFormat(state->ren, 5, 5, "[F1] show info");
    }

    SDL_RenderPresent(state->ren);
    state->needs_redraw = false;
  }

  //SDL_Delay(1);
  return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* s, SDL_Event* event) {
  AppState* state = s;

  switch (event->type) {
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
      int new_width, new_height;
      SDL_GetRenderOutputSize(state->ren, &new_width, &new_height);

      if (new_width != state->width || new_height != state->height) {
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
      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      float mx = event->button.x;
      float my = event->button.y;

      if (state->drag_mode == DRAG_NONE && event->button.button == SDL_BUTTON_LEFT) {
        // check if clicking on a point
        find_hover(state, mx, my);

        // only allow point dragging if editing is allowed
        if (editing_allowed(state)) {
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
      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (event->button.button == SDL_BUTTON_LEFT ||
          event->button.button == SDL_BUTTON_RIGHT) {
        // if we were dragging a coefficient in point cloud mode, recalculate everything
        bool was_dragging_coeff_cloud = (state->drag_mode == DRAG_COEFF &&
                                         state->view_mode == VIEW_MODE_POINT_CLOUD);
        state->drag_mode = DRAG_NONE;
        state->drag_index = -1;

        // full recalculation after drag ends
        if (was_dragging_coeff_cloud) {
          state->drag_skip = 1;
          update_combination_cloud(state, 1);  // skip = 1 means no skipping
          state->needs_redraw = true;
        }
      }
      break;

    case SDL_EVENT_MOUSE_MOTION: {
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
        state->poly->roots[state->drag_index] = cx + cy * I;
        rebuild_from_roots(state);
        state->needs_redraw = true;

      } else if (state->drag_mode == DRAG_COEFF && state->drag_index >= 0) {
        float cx, cy;
        screen_to_complex(state, mx, my, &cx, &cy);

        if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
          state->base_coeffs[state->drag_index] = cx + cy * I;
          // during dragging, skip combinations if there are too many
          size_t skip = 1;
          if (state->num_combinations > DRAG_RENDER_POINTS) {
            // calculate skip to bring total below DRAG_RENDER_POINTS
            skip = (state->num_combinations + DRAG_RENDER_POINTS-1) / DRAG_RENDER_POINTS;
          }
          state->drag_skip = skip;
          update_combination_cloud(state, skip);
        } else {
          state->poly->coeffs[state->drag_index] = cx + cy * I;
          rebuild_from_coeffs(state);
        }
        state->needs_redraw = true;

      } else {
        // update hover state
        DragMode old_mode = state->hover_mode;
        int old_index = state->hover_index;
        find_hover(state, mx, my);
        if (state->hover_mode != old_mode || state->hover_index != old_index) {
          state->needs_redraw = true;
        }
      }
      break;
    }

    case SDL_EVENT_MOUSE_WHEEL: {
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
      break;
    }

    case SDL_EVENT_KEY_UP:

      switch (event->key.scancode) {
        case SCANCODE_PAN_UP:
          state->pan_keys[0] = false;
          break;
        case SCANCODE_PAN_LEFT:
          state->pan_keys[1] = false;
          break;
        case SCANCODE_PAN_DOWN:
          state->pan_keys[2] = false;
          break;
        case SCANCODE_PAN_RIGHT:
          state->pan_keys[3] = false;
          break;
        case SCANCODE_INC_ZOOM:
          state->zoom_keys[0] = false;
          break;
        case SCANCODE_DEC_ZOOM:
          state->zoom_keys[1] = false;
          break;
        default: ;
      }
      break;

    case SDL_EVENT_KEY_DOWN:
      size_t curr_deg = state->poly->degree;
      switch (event->key.scancode) {
        case SCANCODE_RESET_VIEW:
          state->scale = 0.005f;
          state->center_x = 0.0f;
          state->center_y = 0.0f;
          state->needs_redraw = true;
          break;
        case SCANCODE_RESET_POLY:
          if (!editing_allowed(state)) break;
          polynomial_free(state->poly);
          state->poly = create_default_polynomial(curr_deg);
          set_polynomial(state->poly);
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            if (state->base_coeffs) {
              for (int i = 0; i < state->num_base_coeffs;i++) {
                // place coeffs on unit circle
                float angle = i * 2.0f * M_PI / (float)state->num_base_coeffs;
                state->base_coeffs[i] = cexpl(I * angle);
              }
            }
            update_combination_cloud(state, 1);
          }
          state->needs_redraw = true;
          break;
        case SCANCODE_INC_DEGREE:
          if (!editing_allowed(state)) break;
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            size_t new_deg = state->poly_degree_cloud + 1;
            if (int_pow(state->num_base_coeffs, new_deg + 1) > 1000000) {
              SDL_Log("too many combinations");
              break;
            }
            state->poly_degree_cloud = new_deg;
            reallocate_combinations(state);
            update_combination_cloud(state, 1);
          } else {
            curr_deg += (curr_deg < MAX_DEGREE) ? 1 : 0;
            state->poly = create_default_polynomial(curr_deg);
            set_polynomial(state->poly);
          }
          state->needs_redraw = true;
          break;

        case SCANCODE_DEC_DEGREE:
          if (!editing_allowed(state)) break;
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            if (state->poly_degree_cloud > 2) {
              state->poly_degree_cloud--;
              reallocate_combinations(state);
              update_combination_cloud(state, 1);
            }
          } else {
            curr_deg -= (curr_deg > 2) ? 1 : 0;
            state->poly = create_default_polynomial(curr_deg);
            set_polynomial(state->poly);
          }
          state->needs_redraw = true;
          break;
        case SCANCODE_ADD_COEFF:
          if (!editing_allowed(state)) break;
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            size_t new_count = state->num_base_coeffs + 1;
            if (int_pow(new_count, state->poly_degree_cloud + 1) > 1000000) {
              SDL_Log("too many combinations");
              break;
            }
            complex long double* new_coeffs = realloc(state->base_coeffs,
                                                       new_count * sizeof(complex long double));
            if (new_coeffs) {
              state->base_coeffs = new_coeffs;
              state->base_coeffs[state->num_base_coeffs] = 1.0 + 0.0 * I;
              state->num_base_coeffs = new_count;
              reallocate_combinations(state);
              update_combination_cloud(state, 1);
              state->needs_redraw = true;
            }
          }
          break;

        case SCANCODE_REM_COEFF:
          if (!editing_allowed(state)) break;
          if (state->view_mode == VIEW_MODE_POINT_CLOUD && state->num_base_coeffs > 2) {
            state->num_base_coeffs--;
            reallocate_combinations(state);
            update_combination_cloud(state, 1);
            state->needs_redraw = true;
          }
          break;
        case SCANCODE_TOGGLE_MODE:
          if (!editing_allowed(state)) break;
          if (state->view_mode == VIEW_MODE_ROOTS) {
            state->view_mode = VIEW_MODE_POINT_CLOUD;

            if (!state->base_coeffs || int_pow(state->num_base_coeffs, state->poly_degree_cloud + 1) > 1000000) {
              state->num_base_coeffs = 2;
              state->base_coeffs = malloc(state->num_base_coeffs * sizeof(complex long double));
              for (size_t i = 0; i < state->num_base_coeffs; i++) {
                long double angle = ((long double)i * M_PI * 2.0L) / (long double)state->num_base_coeffs;
                state->base_coeffs[i] = cexpl(I * angle);
              }
              state->poly_degree_cloud = 2;
            }

            reallocate_combinations(state);
            update_combination_cloud(state, 1);
          } else {
            state->view_mode = VIEW_MODE_ROOTS;
            if (state->combination_roots) {
              free(state->combination_roots);
              state->combination_roots = nullptr;
            }
            if (state->num_distinct) {
              free(state->num_distinct);
              state->num_distinct = nullptr;
            }
            state->num_combinations = 0;
          }
          state->needs_redraw = true;
          break;
        case SCANCODE_LAPACK:
          if (!editing_allowed(state)) break;
          state->lapack = !state->lapack;
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
        case SCANCODE_PAN_UP:
          state->pan_keys[0] = true;
          break;
        case SCANCODE_PAN_LEFT:
          state->pan_keys[1] = true;
          break;
        case SCANCODE_PAN_DOWN:
          state->pan_keys[2] = true;
          break;
        case SCANCODE_PAN_RIGHT:
          state->pan_keys[3] = true;
          break;
        case SCANCODE_INC_ZOOM:
          state->zoom_keys[0] = true;
          break;
        case SCANCODE_DEC_ZOOM:
          state->zoom_keys[1] = true;
          break;

        case SCANCODE_INC_GAMMA:
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            // adjust point radius in cloud mode
            state->cloud_point_radius += 0.2f;
            if (state->cloud_point_radius > CLOUD_POINT_RADIUS_MAX) {
              state->cloud_point_radius = CLOUD_POINT_RADIUS_MAX;
            }
          } else {
            // adjust gamma in roots mode
            float g = get_gamma();
            set_gamma(g + (g < 3.0f ? 0.05f : 0.0f));
          }
          state->needs_redraw = true;
          break;
        case SCANCODE_DEC_GAMMA:
          if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
            // adjust point radius in cloud mode
            state->cloud_point_radius -= 0.2f;
            if (state->cloud_point_radius < CLOUD_POINT_RADIUS_MIN) {
              state->cloud_point_radius = CLOUD_POINT_RADIUS_MIN;
            }
          } else {
            // adjust gamma in roots mode
            float g = get_gamma();
            set_gamma(g - (g > 0.05f ? 0.05f : 0.f));
          }
          state->needs_redraw = true;
          break;
        // animation controls (always available when animation loaded)
        case SCANCODE_ANIM_PLAY:
          if (state->anim_state) {
            bool playing = !state->anim_state->playing;
            anim_set_playing(state->anim_state, playing);
            state->anim_last_tick = SDL_GetTicks();
            state->needs_redraw = true;
          }
          break;
        case SCANCODE_ANIM_RESET:
          if (state->anim_state) {
            anim_reset(state->anim_state);
            anim_update(state->anim_state, 0.0f);

            // apply initial state
            if (state->view_mode == VIEW_MODE_POINT_CLOUD) {
              for (size_t i = 0; i < state->anim_state->num_points && i < state->num_base_coeffs; i++) {
                state->base_coeffs[i] = state->anim_state->points[i];
              }
              update_combination_cloud(state, 1);
            } else if (state->poly && state->poly->roots_valid) {
              for (size_t i = 0; i < state->anim_state->num_points && i < state->poly->num_distinct_roots; i++) {
                state->poly->roots[i] = state->anim_state->points[i];
              }
              rebuild_from_roots(state);
            }
            state->needs_redraw = true;
          }
          break;
        case SCANCODE_QUIT:
          return SDL_APP_SUCCESS;
        default:
          break;
      }
      break;

    case SDL_EVENT_QUIT:
      return SDL_APP_SUCCESS;

    default:
      break;
  }

  bool panning = false;
  bool zooming = false;
  for (int i = 0; i < sizeof(state->pan_keys) / sizeof(state->pan_keys[0]); i++) {
    panning |= state->pan_keys[i];
  }

  for (int i = 0; i < sizeof(state->zoom_keys) / sizeof(state->zoom_keys[0]); i++) {
    zooming |= state->zoom_keys[i];
  }

  if (!panning) {
    // reset accumulated pan
    state->pan_accel = 1.0f;
    state->pan_repeat = 0;
  } else {
    if (state->pan_keys[0]) {
      state->center_y -= PAN_BASE_RATE*state->pan_accel*state->scale;
    }
    if (state->pan_keys[1]) {
      state->center_x -= PAN_BASE_RATE*state->pan_accel*state->scale;
    }
    if (state->pan_keys[2]) {
      state->center_y += PAN_BASE_RATE*state->pan_accel*state->scale;
    }
    if (state->pan_keys[3]) {
      state->center_x += PAN_BASE_RATE*state->pan_accel*state->scale;
    }
    state->pan_repeat++;
    if (state->pan_repeat > PAN_REPEAT_THRESH && state->pan_accel < PAN_ACCEL_MAX) state->pan_accel *= PAN_ACCEL;
    state->needs_redraw = true;
  }

  if (!zooming) {
    // reset accumulated zoom
    state->zoom_accel = 1.0f;
    state->zoom_repeat = 0;
  } else {
    if (state->zoom_keys[0]) {
      state->scale += ZOOM_BASE_RATE*state->scale*state->zoom_accel;
      if (state->zoom_dir) state->zoom_repeat = 0;
    }
    if (state->zoom_keys[1]) {
      state->scale -= ZOOM_BASE_RATE*state->scale*state->zoom_accel;
      if (!state->zoom_dir) state->zoom_repeat = 0;
    }

    state->zoom_repeat++;
    if (state->zoom_repeat > ZOOM_REPEAT_THRESH && state->zoom_accel < ZOOM_ACCEL_MAX) state->zoom_accel *= ZOOM_ACCEL;
    if (state->scale < 1e-6f) state->scale = 1e-6f;
    if (state->scale > 1.0f) state->scale = 1.0f;
    state->needs_redraw = true;
  }

  return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* s, SDL_AppResult result) {
  AppState* state = s;

  if (result == SDL_APP_FAILURE) {
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "error",
                              SDL_GetError(), state ? state->win : NULL);
  }

  if (state) {
    if (state->anim_state) anim_free_state(state->anim_state);
    if (state->anim_script) anim_free_script(state->anim_script);
    if (state->texture) SDL_DestroyTexture(state->texture);
    if (state->poly) polynomial_free(state->poly);
    if (state->combination_roots) free(state->combination_roots);
    if (state->base_coeffs) free(state->base_coeffs);
    cleanup_renderer();
    SDL_free(state);
  }
}