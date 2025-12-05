//
// Created by Lin Yicheng on 2/12/25.
//

#include "animation.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"

ease_type_t anim_ease_from_string(const char *name) {
  if (!name) return EASE_LINEAR;
  if (strcmp(name, "linear") == 0) return EASE_LINEAR;
  if (strcmp(name, "smooth") == 0) return EASE_SMOOTH;
  if (strcmp(name, "in") == 0) return EASE_IN;
  if (strcmp(name, "out") == 0) return EASE_OUT;
  if (strcmp(name, "inout") == 0) return EASE_IN_OUT;
  if (strcmp(name, "elastic") == 0) return EASE_ELASTIC;
  if (strcmp(name, "bounce") == 0) return EASE_BOUNCE;
  return EASE_LINEAR;
}

const char *anim_ease_to_string(ease_type_t ease) {
  switch (ease) {
    case EASE_LINEAR: return "linear";
    case EASE_SMOOTH: return "smooth";
    case EASE_IN: return "in";
    case EASE_OUT: return "out";
    case EASE_IN_OUT: return "inout";
    case EASE_ELASTIC: return "elastic";
    case EASE_BOUNCE: return "bounce";
  }
  return "linear";
}

float anim_map_ease(ease_type_t ease, float t) {
  if (t <= 0.0f) return 0.0f;
  if (t >= 1.0f) return 1.0f;

  switch (ease) {
    case EASE_LINEAR:
      return t;
    case EASE_SMOOTH:
      return t * t * (3.0f - 2.0f * t); // 3t^3 - 2t^2
    case EASE_IN:
      return t * t; // t^2
    case EASE_OUT:
      return 1.0f - (1.0f - t) * (1.0f - t); // 2t - t^2
    case EASE_IN_OUT:
      if (t < 0.5f)
        return 4.0f * t * t * t; // 4t^3
      return 1.0f - powf(-2.0f * t + 2.0f, 3.0f) / 2.0f; // 1 - (2 - 2t)^3
    case EASE_ELASTIC: {
      float c4 = (2.0f * (float) M_PI) / 3.0f; // frequency
      return powf(2.0f, -10.0f * t) * sinf((t * 10.0f - 0.75f) * c4) + 1.0f; // decay
    }
    case EASE_BOUNCE: {
      // piecewise bouncing height
      float n1 = 7.5625f;
      float d1 = 2.75f;
      if (t < 1.0f / d1)
        return n1 * t * t;
      if (t < 2.0f / d1) {
        t -= 1.5f / d1;
        return n1 * t * t + 0.75f;
      }
      if (t < 2.5f / d1) {
        t -= 2.25f / d1;
        return n1 * t * t + 0.9375f;
      }
      t -= 2.625f / d1;
      return n1 * t * t + 0.984375f;
    }
  }
  return t;
}

// interpolation fns between points
cxldouble anim_interpolate_complex(cxldouble a, cxldouble b, float t) {
  long double re = cxreall(a) + t * (cxreall(b) - cxreall(a));
  long double im = cximagl(a) + t * (cximagl(b) - cximagl(a));
  return CXL(re, im);
}

anim_view_t anim_interpolate_view(anim_view_t a, anim_view_t b, float t) {
  anim_view_t result;
  // interpolate scale logarithmically (canonical solution)
  result.scale = expf(logf(a.scale) + t * (logf(b.scale) - logf(a.scale)));
  result.center_x = a.center_x + t * (b.center_x - a.center_x);
  result.center_y = a.center_y + t * (b.center_y - a.center_y);
  return result;
}

// json parsing with cjson
// helper to get number with default
static double json_get_number(cJSON *obj, const char *key, double def) {
  cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
  if (cJSON_IsNumber(item)) return item->valuedouble;
  return def;
}

// helper to get string with default
static const char *json_get_string(cJSON *obj, const char *key, const char *def) {
  cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
  if (cJSON_IsString(item) && item->valuestring) return item->valuestring;
  return def;
}

// helper to get bool with default
static bool json_get_bool(cJSON *obj, const char *key, bool def) {
  cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
  if (cJSON_IsBool(item)) return cJSON_IsTrue(item);
  return def;
}

// parse a single keyframe
static bool parse_keyframe(cJSON *kf_json, anim_keyframe_t *kf) {
  memset(kf, 0, sizeof(anim_keyframe_t));

  // keyframe time
  kf->time = (float) json_get_number(kf_json, "time", 0.0);

  // easing function
  const char *ease_str = json_get_string(kf_json, "ease", "smooth");
  kf->ease = anim_ease_from_string(ease_str);

  // parse points array: [[re, im], [re, im], ...]
  cJSON *points_json = cJSON_GetObjectItemCaseSensitive(kf_json, "points");
  if (cJSON_IsArray(points_json)) {
    kf->points_valid = true;
    kf->num_points = 0;

    cJSON *pt;
    cJSON_ArrayForEach(pt, points_json) {
      if (kf->num_points >= ANIM_MAX_POINTS) break;

      if (cJSON_IsArray(pt) && cJSON_GetArraySize(pt) >= 2) {
        cJSON *re_item = cJSON_GetArrayItem(pt, 0);
        cJSON *im_item = cJSON_GetArrayItem(pt, 1);

        double re = cJSON_IsNumber(re_item) ? re_item->valuedouble : 0.0;
        double im = cJSON_IsNumber(im_item) ? im_item->valuedouble : 0.0;

        kf->points[kf->num_points++] = CXL((long double) re, (long double) im);
      }
    }
  }

  // parse view object if it exists
  cJSON *view_json = cJSON_GetObjectItemCaseSensitive(kf_json, "view");
  if (cJSON_IsObject(view_json)) {
    kf->view_valid = true;
    kf->view.scale = (float) json_get_number(view_json, "scale", 0.005);
    kf->view.center_x = (float) json_get_number(view_json, "x", 0.0);
    kf->view.center_y = (float) json_get_number(view_json, "y", 0.0);
  }

  return kf->points_valid;
}


anim_script_t *anim_parse_script(const char *json_str) {
  if (!json_str) return NULL;

  cJSON *root = cJSON_Parse(json_str);
  if (!root) {
    const char *error = cJSON_GetErrorPtr();
    if (error)
      fprintf(stderr, "animation: json parse error near: %s\n", error);
    return NULL;
  }

  anim_script_t *script = calloc(1, sizeof(anim_script_t));
  if (!script) {
    cJSON_Delete(root);
    return NULL;
  }

  // desired mode, either point cloud mode or single polynomial roots mode
  const char *mode_str = json_get_string(root, "mode", "cloud");
  script->mode = (strcmp(mode_str, "roots") == 0) ? ANIM_MODE_ROOTS : ANIM_MODE_CLOUD;

  // parse degree and num_coeffs if they exist
  script->degree = (size_t) json_get_number(root, "degree", 4);
  script->num_coeffs = (size_t) json_get_number(root, "num_coeffs", 0);

  // parse loop settings
  script->loop = json_get_bool(root, "loop", false);
  script->loop_delay = (float) json_get_number(root, "loop_delay", 0.0);

  // parse keyframes
  cJSON *keyframes_json = cJSON_GetObjectItemCaseSensitive(root, "keyframes");
  if (!cJSON_IsArray(keyframes_json)) {
    fprintf(stderr, "animation: missing or invalid keyframes array\n");
    free(script);
    cJSON_Delete(root);
    return NULL;
  }

  script->num_keyframes = 0;
  cJSON *kf_json;
  cJSON_ArrayForEach(kf_json, keyframes_json) {
    if (script->num_keyframes >= ANIM_MAX_KEYFRAMES) break;

    if (cJSON_IsObject(kf_json))
      if (parse_keyframe(kf_json, &script->keyframes[script->num_keyframes]))
        script->num_keyframes++;
  }

  if (script->num_keyframes == 0) {
    fprintf(stderr, "animation: no valid keyframes found\n");
    free(script);
    cJSON_Delete(root);
    return NULL;
  }

  // validate keyframe times are monotonically increasing
  for (size_t i = 1; i < script->num_keyframes; i++) {
    if (script->keyframes[i].time < script->keyframes[i - 1].time) {
      fprintf(stderr, "animation: keyframe %zu time (%.3f) is before keyframe %zu time (%.3f)\n",
              i, script->keyframes[i].time, i - 1, script->keyframes[i - 1].time);
      free(script);
      cJSON_Delete(root);
      return NULL;
    }
  }

  // infer num_coeffs from first keyframe if not specified
  if (script->num_coeffs == 0 && script->keyframes[0].points_valid) {
    script->num_coeffs = script->keyframes[0].num_points;
  }

  // calculate the total duration
  script->total_duration = script->keyframes[script->num_keyframes - 1].time;

  cJSON_Delete(root);

  // build sorted index arrays for binary search
  script->num_points_kfs = 0;
  script->num_view_kfs = 0;

  for (size_t i = 0; i < script->num_keyframes; i++) {
    if (script->keyframes[i].points_valid) {
      script->points_kf_indices[script->num_points_kfs++] = i;
    }
    if (script->keyframes[i].view_valid) {
      script->view_kf_indices[script->num_view_kfs++] = i;
    }
  }
  return script;
}

anim_script_t *anim_load_script(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "animation: failed to open %s\n", filename);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (size <= 0 || size > 1024 * 1024) {
    fprintf(stderr, "animation: invalid file size\n");
    fclose(f);
    return NULL;
  }

  char *json_str = malloc(size + 1);
  if (!json_str) {
    fclose(f);
    return NULL;
  }

  if (fread(json_str, 1, size, f) != (size_t) size) {
    fprintf(stderr, "animation: failed to read file\n");
    free(json_str);
    fclose(f);
    return NULL;
  }
  json_str[size] = '\0';
  fclose(f);

  anim_script_t *script = anim_parse_script(json_str);
  free(json_str);
  return script;
}

void anim_free_script(anim_script_t *script) {
  free(script);
}

anim_state_t *anim_create_state(anim_script_t *script) {
  if (!script) return NULL;

  anim_state_t *state = calloc(1, sizeof(anim_state_t));
  if (!state) return NULL;

  state->script = script;
  state->current_time = 0.0f;
  state->playing = false;
  state->finished = false;
  state->current_keyframe = 0;
  state->view_changed = false;

  // initialize from first keyframe
  if (script->num_keyframes > 0) {
    anim_keyframe_t *first = &script->keyframes[0];
    if (first->points_valid) {
      state->num_points = first->num_points;
      memcpy(state->points, first->points, first->num_points * sizeof(cxldouble));
    }
    if (first->view_valid) {
      state->view = first->view;
    } else {
      state->view.scale = 0.005f;
      state->view.center_x = 0.0f;
      state->view.center_y = 0.0f;
    }
  }

  return state;
}

void anim_free_state(anim_state_t *state) {
  free(state);
}

void anim_reset(anim_state_t *state) {
  if (!state) return;
  state->current_time = 0.0f;
  state->current_keyframe = 0;
  state->finished = false;
  state->view_changed = false;
}

void anim_set_playing(anim_state_t *state, bool playing) {
  if (!state) return;
  state->playing = playing;
  if (playing && state->finished) {
    anim_reset(state);
  }
}

// find index of last keyframe with time <= target
static size_t find_keyframe_at_time(anim_script_t *script, float time) {
  if (script->num_keyframes == 0) return 0;

  size_t lo = 0;
  size_t hi = script->num_keyframes;

  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (script->keyframes[mid].time <= time)
      lo = mid + 1;
    else
      hi = mid;
  }

  // lo is first index where time > target, so lo - 1 is what we want
  return (lo == 0) ? 0 : lo - 1;
}

void anim_seek(anim_state_t *state, float time) {
  if (!state || !state->script) return;

  if (time < 0.0f) time = 0.0f;
  if (time > state->script->total_duration) {
    time = state->script->total_duration;
  }

  state->current_time = time;
  state->finished = false;
  state->current_keyframe = find_keyframe_at_time(state->script, time);

  anim_update(state, 0.0f);
}

static size_t find_prev_points_kf(anim_script_t *script, size_t index) {
  size_t i = bsearch_le(script->points_kf_indices, script->num_points_kfs, index);
  return (i == SIZE_MAX) ? 0 : script->points_kf_indices[i];
}

static size_t find_prev_view_kf(anim_script_t *script, size_t index) {
  size_t i = bsearch_le(script->view_kf_indices, script->num_view_kfs, index);
  return (i == SIZE_MAX) ? 0 : script->view_kf_indices[i];
}

static size_t find_next_points_kf(anim_script_t *script, size_t index) {
  size_t i = bsearch_gt(script->points_kf_indices, script->num_points_kfs, index);
  return (i == SIZE_MAX) ? index : script->points_kf_indices[i];
}

static size_t find_next_view_kf(anim_script_t *script, size_t index) {
  size_t i = bsearch_gt(script->view_kf_indices, script->num_view_kfs, index);
  return (i == SIZE_MAX) ? index : script->view_kf_indices[i];
}

// animation update step
bool anim_update(anim_state_t *state, float delta_time) {
  if (!state || !state->script) return false;

  anim_script_t *script = state->script;
  bool changed = false;
  state->view_changed = false;

  // advance time if playing
  if (state->playing && !state->finished) {
    state->current_time += delta_time;
    changed = true;

    if (state->current_time >= script->total_duration) {
      // animation ended, check if we should loop
      if (script->loop) {
        if (state->current_time > (script->total_duration + script->loop_delay)) {
          state->current_time = 0;
          state->current_keyframe = 0;
        }
      } else {
        state->current_time = script->total_duration;
        state->finished = true;
      }
    }
  } else if (state->finished) return false;

  if (script->num_keyframes == 0) return changed;

  // find current segment
  size_t kf_idx = find_keyframe_at_time(state->script, state->current_time);
  state->current_keyframe = kf_idx;

  // interpolate points
  size_t prev_pts = find_prev_points_kf(script, kf_idx);
  size_t next_pts = find_next_points_kf(script, kf_idx);

  if (prev_pts == next_pts) {
    // endpoints
    anim_keyframe_t *kf = &script->keyframes[prev_pts];
    if (kf->points_valid) {
      state->num_points = kf->num_points;
      memcpy(state->points, kf->points, kf->num_points * sizeof(cxldouble));
    }
  } else {
    anim_keyframe_t *kf_a = &script->keyframes[prev_pts];
    anim_keyframe_t *kf_b = &script->keyframes[next_pts];

    float t0 = kf_a->time;
    float t1 = kf_b->time;
    float t = (state->current_time - t0) / (t1 - t0);
    t = anim_map_ease(kf_b->ease, t);

    size_t n = (kf_a->num_points < kf_b->num_points) ? kf_a->num_points : kf_b->num_points;
    state->num_points = n;
    for (size_t i = 0; i < n; i++) {
      state->points[i] = anim_interpolate_complex(kf_a->points[i], kf_b->points[i], t);
    }
  }

  // interpolate view
  size_t prev_view = find_prev_view_kf(script, kf_idx);
  size_t next_view = find_next_view_kf(script, kf_idx);

  anim_view_t old_view = state->view;
  if (prev_view == next_view) {
    // endpoints
    anim_keyframe_t *kf = &script->keyframes[prev_view];
    if (kf->view_valid) {
      state->view = kf->view;
    }
  } else {
    anim_keyframe_t *kf_a = &script->keyframes[prev_view];
    anim_keyframe_t *kf_b = &script->keyframes[next_view];

    float t0 = kf_a->time;
    float t1 = kf_b->time;
    float t = (state->current_time - t0) / (t1 - t0);
    t = anim_map_ease(kf_b->ease, t);

    state->view = anim_interpolate_view(kf_a->view, kf_b->view, t);
  }

  // check if view changed
  // means user has moved the screen
  if (state->view.scale != old_view.scale ||
      state->view.center_x != old_view.center_x ||
      state->view.center_y != old_view.center_y) {
    state->view_changed = true;
  }

  return changed;
}
