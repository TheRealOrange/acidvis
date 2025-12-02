//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_CLOUD_H
#define POLYNOMIAL_CLOUD_H

#include "app.h"

#define DRAG_RENDER_POINTS 20000

// allocate/reallocate combination arrays based on current settings
void cloud_reallocate_combinations(AppState *state);

// update point cloud when coefficients change
// skip parameter: if > 1, only computes every skip-th combination for faster dragging
void cloud_update(AppState *state, size_t skip);

// switch to cloud mode
void cloud_mode_enter(AppState *state);

// exit cloud mode
void cloud_mode_exit(AppState *state);

#endif // POLYNOMIAL_CLOUD_H
