//
// Created by Lin Yicheng on 30/11/25.
//

#ifndef POLYNOMIAL_INPUT_H
#define POLYNOMIAL_INPUT_H

#include <SDL3/SDL.h>
#include "app.h"

// scancode definitions for key bindings
#define SCANCODE_RESET_VIEW    SDL_SCANCODE_R
#define SCANCODE_RESET_POLY    SDL_SCANCODE_C

#define SCANCODE_INC_DEGREE    SDL_SCANCODE_Y
#define SCANCODE_DEC_DEGREE    SDL_SCANCODE_H

#define SCANCODE_ADD_COEFF     SDL_SCANCODE_EQUALS
#define SCANCODE_REM_COEFF     SDL_SCANCODE_MINUS

#define SCANCODE_PAN_UP        SDL_SCANCODE_I
#define SCANCODE_PAN_LEFT      SDL_SCANCODE_J
#define SCANCODE_PAN_DOWN      SDL_SCANCODE_K
#define SCANCODE_PAN_RIGHT     SDL_SCANCODE_L

#define SCANCODE_INC_ZOOM      SDL_SCANCODE_U
#define SCANCODE_DEC_ZOOM      SDL_SCANCODE_O

#define SCANCODE_INC_GAMMA     SDL_SCANCODE_N
#define SCANCODE_DEC_GAMMA     SDL_SCANCODE_M

#define SCANCODE_TOGGLE_MODE   SDL_SCANCODE_T
#define SCANCODE_INVERT        SDL_SCANCODE_V
#define SCANCODE_LOG_SCALE     SDL_SCANCODE_Z

#define SCANCODE_TOGGLE_ROOTS  SDL_SCANCODE_Q
#define SCANCODE_TOGGLE_COEFFS SDL_SCANCODE_W

#define SCANCODE_LAPACK        SDL_SCANCODE_B
#define SCANCODE_TOGGLE_INFO   SDL_SCANCODE_F1
#define SCANCODE_SCALE_MODE    SDL_SCANCODE_S

#define SCANCODE_ANIM_PLAY     SDL_SCANCODE_SPACE
#define SCANCODE_ANIM_RESET    SDL_SCANCODE_X

#define SCANCODE_QUIT          SDL_SCANCODE_ESCAPE

// acceleration parameters for keyboard navigation
#define ZOOM_ACCEL             1.08f
#define ZOOM_BASE_RATE         0.03f
#define ZOOM_ACCEL_MAX         10.0f
#define ZOOM_REPEAT_THRESH     5

#define PAN_ACCEL              1.08f
#define PAN_BASE_RATE          5.0f
#define PAN_ACCEL_MAX          5.0f
#define PAN_REPEAT_THRESH      5

// process SDL event, returns SDL_APP_CONTINUE or SDL_APP_SUCCESS
SDL_AppResult input_handle_event(AppState *state, SDL_Event *event);

// process continuous keyboard state (pan/zoom acceleration)
void input_process_continuous(AppState *state);

// coordinate conversion helpers
void screen_to_complex(AppState *state, float sx, float sy, float *cx, float *cy);
void complex_to_screen(AppState *state, float cx, float cy, float *sx, float *sy);

// hit testing for draggable points
bool input_point_hit_test(AppState *state, float mouse_x, float mouse_y,
                          float complex_x, float complex_y);
void input_find_hover(AppState *state, float mouse_x, float mouse_y);

#endif // POLYNOMIAL_INPUT_H
