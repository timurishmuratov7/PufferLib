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

static float vertical_braking_acceleration(BoosterLanding* env, float mass,
        float vertical_thrust) {
    return mass > 0.0f ? vertical_thrust / mass - env->gravity : 0.0f;
}

// Constant-acceleration coast-then-burn estimate at expected mid-burn mass.
// This is a viability estimate, not a fuel economy objective.
static float estimated_landing_fuel(BoosterLanding* env, float vertical_thrust) {
    float altitude = fmaxf(env->altitude, 0.0f);
    float target_speed = env->max_landing_speed;
    float downward_speed = fmaxf(-env->velocity, 0.0f);

    if (altitude <= 0.0f || env->fuel_burn_rate <= 0.0f || env->gravity <= 0.0f) {
        return 0.0f;
    }

    float effective_mass = env->dry_mass + 0.5f * env->fuel;
    float braking_acceleration = vertical_braking_acceleration(
        env, effective_mass, vertical_thrust);
    if (braking_acceleration <= 0.001f) {
        return env->fuel + env->initial_fuel;
    }

    float denominator = 1.0f / env->gravity + 1.0f / braking_acceleration;
    float numerator = 2.0f * altitude
        + env->velocity * env->velocity / env->gravity
        + target_speed * target_speed / braking_acceleration;
    float switch_speed = sqrtf(fmaxf(numerator / denominator, 0.0f));
    switch_speed = fmaxf(switch_speed, downward_speed);
    float burn_time = fmaxf(switch_speed - target_speed, 0.0f)
        / braking_acceleration;
    return env->fuel_burn_rate * burn_time;
}

static float landing_safety_risk(BoosterLanding* env) {
    float altitude = fmaxf(env->altitude, 0.0f);
    float downward_speed = fmaxf(-env->velocity, 0.0f);
    float vertical_thrust = env->thrust * fmaxf(cosf(env->angle), 0.0f);
    float braking_acceleration = vertical_braking_acceleration(
        env, current_mass(env), vertical_thrust);
    float stopping_distance = 0.0f;

    if (downward_speed > env->max_landing_speed) {
        if (braking_acceleration <= 0.001f) {
            return 1.0f;
        }
        stopping_distance =
            (downward_speed * downward_speed
                - env->max_landing_speed * env->max_landing_speed)
            / (2.0f * braking_acceleration);
    }

    float distance_risk = stopping_distance / fmaxf(altitude, 1.0f);
    float required_fuel = estimated_landing_fuel(env, vertical_thrust);
    float fuel_risk = required_fuel > 0.0f
        ? required_fuel / fmaxf(env->fuel, 0.01f)
        : 0.0f;
    return clampf(fmaxf(distance_risk, fuel_risk), 0.0f, 1.0f);
}

static float landing_safety_potential(BoosterLanding* env) {
    return -landing_safety_risk(env);
}

static float ballistic_touchdown_speed(BoosterLanding* env) {
    float speed_squared = env->velocity * env->velocity
        + 2.0f * env->gravity * fmaxf(env->altitude, 0.0f);
    return sqrtf(fmaxf(speed_squared, 0.0f));
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
    env->actions = (float*)calloc(3, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_init(BoosterLanding* env) {
    allocate(env);
    env->num_agents = 1;
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    env->max_episode_steps = 1200;
    env->rollout_horizon = 128;
    env->rollout_step = 0;

    env->initial_altitude = 2000.0f;
    env->initial_downward_velocity = 35.0f;
    env->reset_altitude_min = 100.0f;
    env->reset_altitude_max = 1500.0f;
    env->reset_downward_velocity_min = 2.0f;
    env->reset_downward_velocity_max = 15.0f;
    env->reset_x_min = -40.0f;
    env->reset_x_max = 40.0f;
    env->reset_angle_min = -0.05f;
    env->reset_angle_max = 0.05f;
    env->dry_mass = 120.0f;
    env->initial_fuel = 100.0f;
    env->gravity = 9.81f;
    env->thrust = 3300.0f;
    env->side_thrust = 80.0f;
    env->fuel_burn_rate = 6.0f;
    env->side_fuel_burn_rate = 0.8f;
    env->moment_arm = 8.0f;
    env->moment_inertia = 3200.0f;
    env->dt = 0.05f;
    env->max_landing_speed = 5.0f;
    env->terminal_reward_scale = 1.0f;
    env->max_landing_x_speed = 8.0f;
    env->max_landing_angle = 0.03f;
    env->max_landing_angular_velocity = 0.05f;
    env->world_width = 420.0f;
    env->landing_pad_width = 160.0f;
    env->max_velocity_obs = 200.0f;
    env->max_x_velocity_obs = 40.0f;
    env->max_angular_velocity_obs = 1.0f;
    env->safety_reward_scale = 0.1f;
    env->reward_gamma = 0.9999f;
    env->client = NULL;
}

void compute_observations(BoosterLanding* env) {
    env->observations[0] = clampf(env->altitude / env->initial_altitude, 0.0f, 1.0f);
    env->observations[1] = clampf(env->velocity / env->max_velocity_obs, -1.0f, 1.0f);
    env->observations[2] = env->world_width > 0.0f ? clampf(env->x / env->world_width, -1.0f, 1.0f) : 0.0f;
    env->observations[3] = clampf(env->x_velocity / env->max_x_velocity_obs, -1.0f, 1.0f);
    env->observations[4] = clampf(env->angle / (float)M_PI, -1.0f, 1.0f);
    env->observations[5] = clampf(env->angular_velocity / env->max_angular_velocity_obs, -1.0f, 1.0f);
    env->observations[6] = env->initial_fuel > 0.0f ?
        clampf(env->fuel / env->initial_fuel, 0.0f, 1.0f) : 0.0f;
    env->observations[7] = landing_safety_risk(env);
}

void c_reset(BoosterLanding* env) {
    env->altitude = rand_float(env, env->reset_altitude_min, env->reset_altitude_max);
    env->start_altitude = env->altitude;
    env->velocity = -rand_float(env,
        env->reset_downward_velocity_min,
        env->reset_downward_velocity_max);
    env->x = rand_float(env, env->reset_x_min, env->reset_x_max);
    env->x_velocity = 0.0f;
    env->angle = rand_float(env, env->reset_angle_min, env->reset_angle_max);
    env->angular_velocity = 0.0f;
    env->fuel = env->initial_fuel;
    env->tick = 0;
    env->awaiting_rollout_reset = false;
    env->terminal_display_ticks = 0;
    env->last_outcome = 0;
    env->episode_return = 0.0f;
    env->episode_main_fuel_used = 0.0f;
    env->episode_side_fuel_used = 0.0f;
    env->episode_main_action_steps = 0.0f;
    env->episode_left_action_steps = 0.0f;
    env->episode_right_action_steps = 0.0f;
    env->episode_simultaneous_side_action_steps = 0.0f;
    compute_observations(env);
}

static void add_log(BoosterLanding* env, bool success, bool timeout,
        bool unrecoverable_fuel) {
    bool landed = env->altitude <= 0.0f;
    bool out_of_bounds = fabsf(env->x) > env->world_width;
    bool on_pad = fabsf(env->x) <= env->landing_pad_width / 2.0f;

    env->log.score += success ? 1.0f : 0.0f;
    env->log.perf += success ? 1.0f : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.crash += (!success && !timeout) ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.start_altitude += env->start_altitude;
    env->log.terminal_fuel += env->fuel;
    env->log.main_fuel_used += env->episode_main_fuel_used;
    env->log.side_fuel_used += env->episode_side_fuel_used;
    float episode_steps = fmaxf((float)env->tick, 1.0f);
    env->log.main_action_rate += env->episode_main_action_steps / episode_steps;
    env->log.left_action_rate += env->episode_left_action_steps / episode_steps;
    env->log.right_action_rate += env->episode_right_action_steps / episode_steps;
    env->log.simultaneous_side_action_rate +=
        env->episode_simultaneous_side_action_steps / episode_steps;
    env->log.landed += landed ? 1.0f : 0.0f;
    if (landed) {
        env->log.touchdown_vertical_speed += fabsf(env->velocity);
        env->log.touchdown_horizontal_speed += fabsf(env->x_velocity);
        env->log.touchdown_angle += fabsf(env->angle);
        env->log.touchdown_angular_velocity += fabsf(env->angular_velocity);
        env->log.failure_off_pad += (!success && !on_pad) ? 1.0f : 0.0f;
        env->log.failure_vertical_speed +=
            (!success && fabsf(env->velocity) > env->max_landing_speed) ? 1.0f : 0.0f;
        env->log.failure_horizontal_speed +=
            (!success && fabsf(env->x_velocity) > env->max_landing_x_speed) ? 1.0f : 0.0f;
        env->log.failure_angle +=
            (!success && fabsf(env->angle) > env->max_landing_angle) ? 1.0f : 0.0f;
        env->log.failure_angular_velocity +=
            (!success && fabsf(env->angular_velocity) > env->max_landing_angular_velocity) ? 1.0f : 0.0f;
    }
    env->log.failure_out_of_bounds += (!success && out_of_bounds) ? 1.0f : 0.0f;
    env->log.failure_fuel_empty += (!success && env->fuel <= 0.001f) ? 1.0f : 0.0f;
    env->log.failure_unrecoverable_fuel += unrecoverable_fuel ? 1.0f : 0.0f;

    if (env->start_altitude < 500.0f) {
        env->log.episodes_100_500 += 1.0f;
        env->log.successes_100_500 += success ? 1.0f : 0.0f;
    } else if (env->start_altitude < 750.0f) {
        env->log.episodes_500_750 += 1.0f;
        env->log.successes_500_750 += success ? 1.0f : 0.0f;
    } else if (env->start_altitude < 1000.0f) {
        env->log.episodes_750_1000 += 1.0f;
        env->log.successes_750_1000 += success ? 1.0f : 0.0f;
    } else if (env->start_altitude < 1250.0f) {
        env->log.episodes_1000_1250 += 1.0f;
        env->log.successes_1000_1250 += success ? 1.0f : 0.0f;
    } else if (env->start_altitude < 1500.0f) {
        env->log.episodes_1250_1500 += 1.0f;
        env->log.successes_1250_1500 += success ? 1.0f : 0.0f;
    } else {
        env->log.episodes_1500_plus += 1.0f;
        env->log.successes_1500_plus += success ? 1.0f : 0.0f;
    }
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

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    bool rollout_boundary = false;
    if (env->rollout_horizon > 0) {
        env->rollout_step += 1;
        if (env->rollout_step >= env->rollout_horizon) {
            env->rollout_step = 0;
            rollout_boundary = true;
        }
    }

    // Native MinGRU state is reset at rollout boundaries. Keep completed
    // episodes idle so a new episode always starts with fresh recurrent state.
    if (env->awaiting_rollout_reset) {
        if (rollout_boundary) {
            c_reset(env);
        }
        return;
    }

    float previous_safety_potential = landing_safety_potential(env);

    bool main_command = (int)env->actions[0] == BURN;
    bool left_command = (int)env->actions[1] == SIDE_FIRE;
    bool right_command = (int)env->actions[2] == SIDE_FIRE;
    env->episode_main_action_steps += main_command ? 1.0f : 0.0f;
    env->episode_left_action_steps += left_command ? 1.0f : 0.0f;
    env->episode_right_action_steps += right_command ? 1.0f : 0.0f;
    env->episode_simultaneous_side_action_steps +=
        left_command && right_command ? 1.0f : 0.0f;

    bool burning = main_command && env->fuel > 0.0f;
    bool left_thruster = left_command && env->fuel > 0.0f;
    bool right_thruster = right_command && env->fuel > 0.0f;
    float main_accel = 0.0f;
    float side_accel = 0.0f;
    float angular_accel = 0.0f;

    if (burning) {
        float fuel_used = fminf(env->fuel, env->fuel_burn_rate * env->dt);
        main_accel = env->thrust / current_mass(env);
        env->fuel -= fuel_used;
        env->episode_main_fuel_used += fuel_used;
    }

    if (left_thruster || right_thruster) {
        int left = left_thruster ? 1 : 0;
        int right = right_thruster ? 1 : 0;
        float fuel_used = fminf(env->fuel, env->side_fuel_burn_rate * env->dt * (left + right));
        env->fuel -= fuel_used;
        env->episode_side_fuel_used += fuel_used;
        side_accel = env->side_thrust * (left - right) / current_mass(env);
        angular_accel = env->side_thrust * env->moment_arm * (left - right) / env->moment_inertia;
    }

    float vertical_acceleration = main_accel * cosf(env->angle) - env->gravity;
    float horizontal_acceleration = main_accel * sinf(env->angle) + side_accel;
    env->velocity += vertical_acceleration * env->dt;
    env->x_velocity += horizontal_acceleration * env->dt;
    env->angular_velocity += angular_accel * env->dt;
    env->angular_velocity *= 0.990f;
    env->angular_velocity = clampf(env->angular_velocity, -env->max_angular_velocity_obs, env->max_angular_velocity_obs);
    env->altitude += env->velocity * env->dt;
    env->x += env->x_velocity * env->dt;
    env->angle += env->angular_velocity * env->dt;
    env->tick += 1;

    bool landed = env->altitude <= 0.0f;
    bool out_of_bounds = fabsf(env->x) > env->world_width;
    bool fuel_exhausted = env->fuel <= 0.001f;
    bool unrecoverable_fuel = !landed && fuel_exhausted
        && ballistic_touchdown_speed(env) > env->max_landing_speed;
    bool timeout = !landed && !out_of_bounds && !unrecoverable_fuel
        && env->tick >= env->max_episode_steps;
    bool on_pad = fabsf(env->x) <= env->landing_pad_width / 2.0f;
    bool stable_attitude = fabsf(env->angle) <= env->max_landing_angle &&
        fabsf(env->angular_velocity) <= env->max_landing_angular_velocity;
    bool stable_horizontal = fabsf(env->x_velocity) <= env->max_landing_x_speed;
    bool stable_vertical = fabsf(env->velocity) <= env->max_landing_speed;
    bool valid_touchdown = landed && on_pad && stable_horizontal && stable_attitude;
    bool success = valid_touchdown && stable_vertical;
    bool done = landed || timeout || out_of_bounds || unrecoverable_fuel;

    float next_safety_potential = done ? 0.0f : landing_safety_potential(env);
    env->rewards[0] = env->safety_reward_scale
        * (env->reward_gamma * next_safety_potential
            - previous_safety_potential);

    if (done) {
        env->rewards[0] += success
            ? env->terminal_reward_scale
            : -env->terminal_reward_scale;
    }

    env->rewards[0] = clampf(env->rewards[0], -1.0f, 1.0f);
    env->episode_return += env->rewards[0];

    if (done) {
        env->altitude = fmaxf(env->altitude, 0.0f);
        env->terminals[0] = 1.0f;
        add_log(env, success, timeout, unrecoverable_fuel);
        compute_observations(env);

        if (env->client != NULL) {
            env->terminal_display_ticks = 90;
            env->last_outcome = success ? 1 : (timeout ? 3 : 2);
            env->last_touchdown_velocity = env->velocity;
            return;
        }

        if (env->rollout_horizon > 0 && !rollout_boundary) {
            env->awaiting_rollout_reset = true;
        } else {
            c_reset(env);
        }
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

    float render_altitude = fmaxf(env->initial_altitude, env->reset_altitude_max);
    float visible_altitude = clampf(env->altitude / render_altitude, 0.0f, 1.0f);
    int rocket_x = env->width / 2 + (int)(clampf(env->x / env->world_width, -1.0f, 1.0f) * (env->width * 0.38f));
    int pad_y = ground_y - 8;
    int rocket_y = pad_y - (int)(visible_altitude * (pad_y - 130));

    Color body = (Color){218, 224, 229, 255};
    float body_width = 24.0f;
    float body_height = 96.0f;
    float half_width = body_width / 2.0f;
    float s = sinf(env->angle);
    float c = cosf(env->angle);
    Vector2 up = (Vector2){s, -c};
    Vector2 right = (Vector2){c, s};
    Vector2 bottom_center = (Vector2){rocket_x, rocket_y};
    Vector2 top_center = (Vector2){
        bottom_center.x + up.x * body_height,
        bottom_center.y + up.y * body_height,
    };
    Vector2 bottom_left = (Vector2){
        bottom_center.x - right.x * half_width,
        bottom_center.y - right.y * half_width,
    };
    Vector2 bottom_right = (Vector2){
        bottom_center.x + right.x * half_width,
        bottom_center.y + right.y * half_width,
    };
    Vector2 top_left = (Vector2){
        top_center.x - right.x * half_width,
        top_center.y - right.y * half_width,
    };
    Vector2 top_right = (Vector2){
        top_center.x + right.x * half_width,
        top_center.y + right.y * half_width,
    };
    DrawTriangle(top_left, bottom_left, bottom_right, body);
    DrawTriangle(top_left, bottom_right, top_right, body);

    bool burning = (int)env->actions[0] == BURN && env->fuel > 0.0f;
    float mass = current_mass(env);
    float fuel_pct = env->initial_fuel > 0.0f ? 100.0f * env->fuel / env->initial_fuel : 0.0f;
    float speed = fabsf(env->velocity);
    Color panel = (Color){12, 18, 28, 220};
    Color muted = (Color){175, 186, 198, 255};
    Color ok = (Color){92, 220, 132, 255};
    Color warn = (Color){255, 185, 70, 255};
    Color speed_color = speed <= env->max_landing_speed ? ok : warn;

    DrawRectangle(8, 8, 260, 254, panel);
    DrawRectangleLines(8, 8, 260, 254, (Color){78, 92, 112, 255});
    DrawText("BOOSTER TELEMETRY", 18, 18, 16, RAYWHITE);
    DrawText(TextFormat("Altitude       %7.1f m", env->altitude), 18, 44, 16, RAYWHITE);
    DrawText(TextFormat("Velocity       %7.1f m/s", env->velocity), 18, 66, 16, speed_color);
    DrawText(TextFormat("X offset       %7.1f m", env->x), 18, 88, 16, RAYWHITE);
    DrawText(TextFormat("X velocity     %7.1f m/s", env->x_velocity), 18, 110, 16, RAYWHITE);
    DrawText(TextFormat("Tilt           %7.1f deg", env->angle * 180.0f / (float)M_PI), 18, 132, 16, RAYWHITE);
    DrawText(TextFormat("Angular vel    %7.2f rad/s", env->angular_velocity), 18, 154, 16, RAYWHITE);
    DrawText(TextFormat("Fuel           %7.1f kg", env->fuel), 18, 176, 16, RAYWHITE);
    DrawText(TextFormat("Fuel remaining %6.1f%%", fuel_pct), 18, 198, 16, RAYWHITE);
    DrawText(TextFormat("Mass           %7.1f kg", mass), 18, 220, 16, RAYWHITE);
    DrawText(TextFormat("Engine         %s", burning ? "BURN" : "COAST"), 18, 242, 16, burning ? warn : muted);

    DrawRectangle(8, env->height - 54, 220, 42, panel);
    DrawText(TextFormat("Step %d / %d", env->tick, env->max_episode_steps), 18, env->height - 46, 16, RAYWHITE);
    DrawText("SPACE burn | A/D top thrusters", 18, env->height - 26, 16, muted);

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
