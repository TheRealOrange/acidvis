//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_CLOUD_H
#define POLYNOMIAL_CLOUD_H

#include "app.h"

#define DRAG_RENDER_POINTS 40000

// allocate/reallocate combination arrays based on current settings
void cloud_reallocate_combinations(AppState *state);

// update point cloud when coefficients change (full solve, no caching)
// skip parameter: if > 1, only computes every skip-th combination
void cloud_update(AppState *state, size_t skip);

// update point cloud during drag (uses incremental solving for speed)
// skip parameter: if > 1, only computes every skip-th combination
void cloud_update_drag(AppState *state, size_t skip);

// call after drag ends to do a full recalculation
void cloud_update_drag_end(AppState *state);

// switch to cloud mode
void cloud_mode_enter(AppState *state);

// exit cloud mode
void cloud_mode_exit(AppState *state);

#endif // POLYNOMIAL_CLOUD_H