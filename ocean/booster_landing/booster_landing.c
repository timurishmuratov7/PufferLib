#include "booster_landing.h"
#include "env.c"

int main() {
    BoosterLanding env = {0};
    c_init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyDown(KEY_SPACE) ? BURN : COAST;
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
