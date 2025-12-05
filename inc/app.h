//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_APP_H
#define POLYNOMIAL_APP_H

#include <SDL3/SDL.h>
#include "compat_complex.h"

#include "animation.h"
#include "polynomial.h"

#define MAX_DEGREE 26

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
  SDL_Window *win;
  SDL_Renderer *ren;
  SDL_Texture *texture;
  polynomial_t *poly;

  // view parameters
  float scale;      // units per pixel in complex plane
  float center_x;   // center of view in complex plane
  float center_y;

  // view mode
  ViewMode view_mode;

  bool lapack;
  bool show_info;
  bool fixed_point_scale;   // true = fixed radius, false = scales with zoom
  float cloud_point_radius;
  size_t drag_skip;         // current skip factor during drag
  bool show_roots;          // show root point markers
  bool show_coeffs;         // show coefficient point markers

  // base coefficients for VIEW_MODE_POINT_CLOUD (the N draggable points)
  cxldouble *base_coeffs;
  size_t num_base_coeffs;
  size_t poly_degree_cloud;

  // combination cloud roots
  cxldouble *combination_roots;
  bool *combination_valid;
  size_t comb_roots_stride;
  size_t num_combinations;  // N^(M+1)
  size_t found_roots;

  // caching for incremental solving during drag
  cxldouble *prev_base_coeffs;  // previous coefficients for delta computation
  int *since_last_update;       // tracks when each combination was last fully solved

  // drag state
  DragMode drag_mode;
  int drag_index;           // which root or coeff is being dragged
  float drag_start_mouse_x;
  float drag_start_mouse_y;
  float drag_start_center_x;
  float drag_start_center_y;

  // keyboard zoom
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
  bool anim_active;               // true when animation is loaded
  char anim_filename[512];        // for reload functionality
  Uint64 anim_last_tick;          // delta time tracking
} AppState;

// initialization and cleanup
bool app_init(AppState *state, int argc, char **argv);
void app_cleanup(AppState *state);

// main loop functions
void app_iterate(AppState *state);

// polynomial management
polynomial_t *app_create_default_polynomial(size_t degree);
void app_rebuild_from_roots(AppState *state);
void app_rebuild_from_coeffs(AppState *state);

// animation
bool app_load_animation(AppState *state, const char *filename);
void app_unload_animation(AppState *state);
void app_update_animation(AppState *state);
bool app_editing_allowed(AppState *state);

#endif // POLYNOMIAL_APP_H