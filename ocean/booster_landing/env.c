#include "booster_landing.h"

static float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static float current_mass(BoosterLanding* env) {
    return env->dry_mass + env->fuel;
}

static float rand_float(BoosterLanding* env, float low, float high) {
    if (high <= low) {
        return low;
    }

    float t = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return low + t * (high - low);
}

void allocate(BoosterLanding* env) {
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_init(BoosterLanding* env) {
    allocate(env);
    env->num_agents = 1;
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    env->max_episode_steps = 600;

    env->initial_altitude = 1000.0f;
    env->initial_downward_velocity = 35.0f;
    env->reset_altitude_min = 50.0f;
    env->reset_altitude_max = 1000.0f;
    env->reset_downward_velocity_min = 5.0f;
    env->reset_downward_velocity_max = 35.0f;
    env->dry_mass = 120.0f;
    env->initial_fuel = 80.0f;
    env->gravity = 9.81f;
    env->thrust = 3000.0f;
    env->fuel_burn_rate = 6.0f;
    env->dt = 0.05f;
    env->max_landing_speed = 5.0f;
    env->landing_pad_width = 80.0f;
    env->max_velocity_obs = 80.0f;
    env->client = NULL;
}

void compute_observations(BoosterLanding* env) {
    float initial_mass = env->dry_mass + env->initial_fuel;
    env->observations[0] = clampf(env->altitude / env->initial_altitude, 0.0f, 1.0f);
    env->observations[1] = clampf(env->velocity / env->max_velocity_obs, -1.0f, 1.0f);
    env->observations[2] = env->initial_fuel > 0.0f ? clampf(env->fuel / env->initial_fuel, 0.0f, 1.0f) : 0.0f;
    env->observations[3] = initial_mass > 0.0f ? clampf(current_mass(env) / initial_mass, 0.0f, 1.0f) : 0.0f;
}

void c_reset(BoosterLanding* env) {
    env->altitude = rand_float(env, env->reset_altitude_min, env->reset_altitude_max);
    env->velocity = -rand_float(env,
        env->reset_downward_velocity_min,
        env->reset_downward_velocity_max);
    env->fuel = env->initial_fuel;
    env->tick = 0;
    env->terminal_display_ticks = 0;
    env->last_outcome = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

static void add_log(BoosterLanding* env, bool success, bool timeout) {
    env->log.score += success ? 1.0f : 0.0f;
    env->log.perf = success ? 1.0f : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.crash += (!success && !timeout) ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_step(BoosterLanding* env) {
    if (env->terminal_display_ticks > 0) {
        env->terminal_display_ticks -= 1;
        env->rewards[0] = 0.0f;
        env->terminals[0] = 1.0f;
        if (env->terminal_display_ticks == 0) {
            c_reset(env);
        }
        return;
    }

    env->rewards[0] = -0.01f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    bool burning = action == BURN && env->fuel > 0.0f;
    float thrust_accel = 0.0f;
    if (burning) {
        float fuel_used = fminf(env->fuel, env->fuel_burn_rate * env->dt);
        thrust_accel = env->thrust / current_mass(env);
        env->fuel -= fuel_used;
    }

    float acceleration = thrust_accel - env->gravity;
    env->velocity += acceleration * env->dt;
    env->altitude += env->velocity * env->dt;
    env->tick += 1;

    bool landed = env->altitude <= 0.0f;
    bool timeout = env->tick >= env->max_episode_steps;
    bool success = landed && fabsf(env->velocity) <= env->max_landing_speed;
    bool done = landed || timeout;

    if (success) {
        env->rewards[0] += 100.0f;
    } else if (landed) {
        env->rewards[0] -= 100.0f + fabsf(env->velocity);
    } else if (timeout) {
        env->rewards[0] -= 100.0f;
    }

    env->episode_return += env->rewards[0];

    if (done) {
        env->altitude = fmaxf(env->altitude, 0.0f);
        env->terminals[0] = 1.0f;
        add_log(env, success, timeout);
        compute_observations(env);

        if (env->client != NULL) {
            env->terminal_display_ticks = 90;
            env->last_outcome = success ? 1 : (timeout ? 3 : 2);
            env->last_touchdown_velocity = env->velocity;
            return;
        }

        c_reset(env);
        return;
    }

    compute_observations(env);
}

Client* make_client(BoosterLanding* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(env->width * RENDER_SCALE, env->height * RENDER_SCALE, "PufferLib Booster Landing");
    SetTargetFPS(60);
    client->initialized = true;
    return client;
}

void c_render(BoosterLanding* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){8, 12, 20, 255});

    int ground_y = env->height - 70;
    DrawRectangle(0, ground_y, env->width, env->height - ground_y, (Color){45, 48, 52, 255});
    DrawRectangle(env->width / 2 - env->landing_pad_width / 2, ground_y - 8,
        env->landing_pad_width, 8, (Color){210, 210, 210, 255});

    float visible_altitude = clampf(env->altitude / env->initial_altitude, 0.0f, 1.0f);
    int rocket_x = env->width / 2;
    int pad_y = ground_y - 8;
    int rocket_y = pad_y - (int)(visible_altitude * (pad_y - 90));

    Color body = (Color){218, 224, 229, 255};
    Color body_shadow = (Color){168, 176, 184, 255};
    Color dark = (Color){38, 46, 56, 255};

    DrawRectangle(rocket_x - 12, rocket_y - 92, 24, 92, body);
    DrawRectangle(rocket_x + 5, rocket_y - 88, 7, 84, body_shadow);
    DrawRectangle(rocket_x - 12, rocket_y - 92, 24, 9, dark);
    DrawRectangle(rocket_x - 12, rocket_y - 66, 24, 3, dark);
    DrawRectangle(rocket_x - 12, rocket_y - 34, 24, 3, dark);

    DrawRectangle(rocket_x - 22, rocket_y - 82, 10, 8, dark);
    DrawRectangle(rocket_x + 12, rocket_y - 82, 10, 8, dark);
    DrawLine(rocket_x - 20, rocket_y - 78, rocket_x - 14, rocket_y - 78, body_shadow);
    DrawLine(rocket_x + 14, rocket_y - 78, rocket_x + 20, rocket_y - 78, body_shadow);

    DrawRectangle(rocket_x - 8, rocket_y - 4, 6, 8, dark);
    DrawRectangle(rocket_x + 2, rocket_y - 4, 6, 8, dark);

    if ((int)env->actions[0] == BURN && env->fuel > 0.0f) {
        DrawTriangle(
            (Vector2){rocket_x - 10, rocket_y + 4},
            (Vector2){rocket_x + 10, rocket_y + 4},
            (Vector2){rocket_x, rocket_y + 42},
            (Color){255, 150, 35, 255}
        );
        DrawTriangle(
            (Vector2){rocket_x - 5, rocket_y + 4},
            (Vector2){rocket_x + 5, rocket_y + 4},
            (Vector2){rocket_x, rocket_y + 28},
            (Color){255, 230, 120, 255}
        );
    }

    bool burning = (int)env->actions[0] == BURN && env->fuel > 0.0f;
    float mass = current_mass(env);
    float fuel_pct = env->initial_fuel > 0.0f ? 100.0f * env->fuel / env->initial_fuel : 0.0f;
    float speed = fabsf(env->velocity);
    Color panel = (Color){12, 18, 28, 220};
    Color muted = (Color){175, 186, 198, 255};
    Color ok = (Color){92, 220, 132, 255};
    Color warn = (Color){255, 185, 70, 255};
    Color speed_color = speed <= env->max_landing_speed ? ok : warn;

    DrawRectangle(8, 8, 220, 188, panel);
    DrawRectangleLines(8, 8, 220, 188, (Color){78, 92, 112, 255});
    DrawText("BOOSTER TELEMETRY", 18, 18, 16, RAYWHITE);
    DrawText(TextFormat("Altitude       %7.1f m", env->altitude), 18, 44, 16, RAYWHITE);
    DrawText(TextFormat("Velocity       %7.1f m/s", env->velocity), 18, 66, 16, speed_color);
    DrawText(TextFormat("Speed limit    %7.1f m/s", env->max_landing_speed), 18, 88, 16, muted);
    DrawText(TextFormat("Fuel           %7.1f kg", env->fuel), 18, 110, 16, RAYWHITE);
    DrawText(TextFormat("Fuel remaining %6.1f%%", fuel_pct), 18, 132, 16, RAYWHITE);
    DrawText(TextFormat("Mass           %7.1f kg", mass), 18, 154, 16, RAYWHITE);
    DrawText(TextFormat("Engine         %s", burning ? "BURN" : "COAST"), 18, 176, 16, burning ? warn : muted);

    DrawRectangle(8, env->height - 54, 220, 42, panel);
    DrawText(TextFormat("Step %d / %d", env->tick, env->max_episode_steps), 18, env->height - 46, 16, RAYWHITE);
    DrawText("SPACE: burn", 18, env->height - 26, 16, muted);

    if (env->terminal_display_ticks > 0) {
        const char* outcome = env->last_outcome == 1 ? "LANDED" :
            (env->last_outcome == 2 ? "CRASHED" : "TIMEOUT");
        Color outcome_color = env->last_outcome == 1 ? ok : (Color){235, 70, 70, 255};
        DrawRectangle(58, env->height / 2 - 48, 244, 92, (Color){8, 12, 20, 235});
        DrawRectangleLines(58, env->height / 2 - 48, 244, 92, outcome_color);
        DrawText(outcome, 82, env->height / 2 - 34, 30, outcome_color);
        DrawText(TextFormat("Touchdown %.1f m/s", env->last_touchdown_velocity),
            82, env->height / 2 + 6, 18, RAYWHITE);
    }

    EndDrawing();
}

void c_close(BoosterLanding* env) {
    if (env->client != NULL) {
        free(env->client);
        env->client = NULL;
    }

    if (IsWindowReady()) {
        CloseWindow();
    }
}
