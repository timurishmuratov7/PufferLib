#include "booster_landing_continuous.h"
#include "env.c"

int main() {
    BoosterLanding env = {0};
    c_init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyDown(KEY_SPACE)
            ? MANUAL_RAW_ACTION_LIMIT : -MANUAL_RAW_ACTION_LIMIT;
        env.actions[1] = 0.0f;
        if (IsKeyDown(KEY_A) && !IsKeyDown(KEY_D)) {
            env.actions[1] = MANUAL_RAW_ACTION_LIMIT;
        } else if (IsKeyDown(KEY_D) && !IsKeyDown(KEY_A)) {
            env.actions[1] = -MANUAL_RAW_ACTION_LIMIT;
        }
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
