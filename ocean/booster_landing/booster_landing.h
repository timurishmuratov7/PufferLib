#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "raylib.h"

#define OBS_SIZE 4
#define COAST 0
#define BURN 1

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 720
#define RENDER_SCALE 1.0f

typedef struct {
    float score;
    float perf;
    float episode_return;
    float episode_length;
    float success;
    float crash;
    float timeout;
    float n;
} Log;

typedef struct {
    bool initialized;
} Client;

typedef struct BoosterLanding {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned int rng;

    int num_agents;
    int width;
    int height;
    int tick;
    int max_episode_steps;
    int terminal_display_ticks;
    int last_outcome;

    float altitude;
    float velocity;
    float fuel;
    float initial_altitude;
    float initial_downward_velocity;
    float reset_altitude_min;
    float reset_altitude_max;
    float reset_downward_velocity_min;
    float reset_downward_velocity_max;
    float dry_mass;
    float initial_fuel;
    float gravity;
    float thrust;
    float fuel_burn_rate;
    float dt;
    float max_landing_speed;
    float landing_pad_width;
    float max_velocity_obs;
    float episode_return;
    float last_touchdown_velocity;

    Client* client;
} BoosterLanding;

void allocate(BoosterLanding* env);
void c_init(BoosterLanding* env);
void c_reset(BoosterLanding* env);
void c_step(BoosterLanding* env);
void c_render(BoosterLanding* env);
void c_close(BoosterLanding* env);
void compute_observations(BoosterLanding* env);
