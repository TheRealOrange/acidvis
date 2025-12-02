//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_UI_H
#define POLYNOMIAL_UI_H

#include <SDL3/SDL.h>
#include "app.h"

#define POINT_RADIUS             10.0f
#define POINT_HIT_RADIUS         12.0f
#define CLOUD_POINT_RADIUS_DEFAULT 1.0f
#define CLOUD_POINT_RADIUS_MIN   0.2f
#define CLOUD_POINT_RADIUS_MAX   10.0f

// draw a filled circle
void ui_draw_circle(SDL_Renderer *ren, float cx, float cy, float r);

// draw a circle outline
void ui_draw_circle_outline(SDL_Renderer *ren, float cx, float cy, float r, int segments);

// render the info overlay
void ui_render_info_overlay(AppState *state);

// render coefficient/root point markers
void ui_render_point_markers(AppState *state);

// render the complete frame
void ui_render_frame(AppState *state);

#endif // POLYNOMIAL_UI_H
