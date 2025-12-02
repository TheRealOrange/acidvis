//
// Created by Lin Yicheng on 2/12/25.
//

#ifndef POLYNOMIAL_ANIMATION_H
#define POLYNOMIAL_ANIMATION_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>

#define ANIM_MAX_POINTS    64
#define ANIM_MAX_KEYFRAMES 256

// easing function types
typedef enum {
  EASE_LINEAR,      // constant speed
  EASE_SMOOTH,      // smooth step (cubic hermite)
  EASE_IN,          // accelerate from zero velocity
  EASE_OUT,         // decelerate to zero velocity
  EASE_IN_OUT,      // accelerate then decelerate
  EASE_ELASTIC,     // overshoot with spring effect
  EASE_BOUNCE       // bounce at destination
} ease_type_t;

typedef enum {
  ANIM_MODE_CLOUD,  // animate base_coeffs in cloud mode
  ANIM_MODE_ROOTS   // animate roots in roots mode
} anim_mode_t;

typedef struct {
  float scale;
  float center_x;
  float center_y;
} anim_view_t;

// single keyframe in an animation
typedef struct {
  float time;                                    // time in seconds
  complex long double points[ANIM_MAX_POINTS];   // coefficient or root positions
  size_t num_points;                             // how many points defined
  bool points_valid;                             // whether points are specified

  anim_view_t view;                              // camera state
  bool view_valid;                               // whether view is specified

  ease_type_t ease;                              // easing to this keyframe
} anim_keyframe_t;

// complete animation specs
typedef struct {
  anim_mode_t mode;
  size_t degree;                                 // polynomial degree (for cloud mode)
  size_t num_coeffs;                             // number of base coefficients
  bool loop;                                     // whether to loop
  float loop_delay;                              // delay before restart (seconds)

  anim_keyframe_t keyframes[ANIM_MAX_KEYFRAMES];
  size_t num_keyframes;

  float total_duration;                          // computed from keyframes

  // computed when parsing the script for
  // quicker searching for keyframes
  // sorted indices of keyframes with valid points/view
  size_t points_kf_indices[ANIM_MAX_KEYFRAMES];
  size_t num_points_kfs;
  size_t view_kf_indices[ANIM_MAX_KEYFRAMES];
  size_t num_view_kfs;
} anim_script_t;

// playback state
typedef struct {
  anim_script_t *script;
  float current_time;
  bool playing;
  bool finished;
  size_t current_keyframe;

  // interpolated output
  complex long double points[ANIM_MAX_POINTS];
  size_t num_points;
  anim_view_t view;
  bool view_changed;                             // true if view updated this frame
} anim_state_t;

// load/parse animation script from a json file
anim_script_t *anim_load_script(const char *filename);
anim_script_t *anim_parse_script(const char *json_str);

void anim_free_script(anim_script_t *script);

// animation state control
anim_state_t *anim_create_state(anim_script_t *script);
void anim_set_playing(anim_state_t *state, bool playing);
void anim_seek(anim_state_t *state, float time);
void anim_reset(anim_state_t *state);

// advance animation by delta_time seconds
// returns true if animation state changed
bool anim_update(anim_state_t *state, float delta_time);

void anim_free_state(anim_state_t *state);

ease_type_t anim_ease_from_string(const char *name);
const char *anim_ease_to_string(ease_type_t ease);
float anim_map_ease(ease_type_t ease, float t);
complex long double anim_interpolate_complex(complex long double a, complex long double b, float t);
anim_view_t anim_interpolate_view(anim_view_t a, anim_view_t b, float t);

#endif //POLYNOMIAL_ANIMATION_H