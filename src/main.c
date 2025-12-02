//
// Created by Lin Yicheng on 30/11/25.
//

#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>
#include <SDL3/SDL.h>

#include "app.h"
#include "input.h"
#include "ui.h"

SDL_AppResult SDL_AppInit(void **s, int argc, char **argv) {
  AppState *state = SDL_calloc(1, sizeof(AppState));
  if (!state) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "failed to allocate app state");
    return SDL_APP_FAILURE;
  }

  *s = state;

  if (!app_init(state, argc, argv)) {
    return SDL_APP_FAILURE;
  }

  return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void *s) {
  AppState *state = s;

  // update animation if active
  app_update_animation(state);

  // process continuous input (keyboard pan/zoom acceleration)
  input_process_continuous(state);

  // render if needed
  if (state->needs_redraw) {
    ui_render_frame(state);
  }

  return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *s, SDL_Event *event) {
  AppState *state = s;
  SDL_AppResult result = input_handle_event(state, event);

  // process continuous input after event handling
  input_process_continuous(state);

  return result;
}

void SDL_AppQuit(void *s, SDL_AppResult result) {
  AppState *state = s;

  if (result == SDL_APP_FAILURE) {
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "error",
                             SDL_GetError(), state ? state->win : nullptr);
  }

  if (state) {
    app_cleanup(state);
    SDL_free(state);
  }
}
