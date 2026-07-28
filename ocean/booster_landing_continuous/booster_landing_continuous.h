#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "raylib.h"

#define OBS_SIZE 8
#define MANUAL_RAW_ACTION_LIMIT 8.0f

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 720
#define RENDER_SCALE 1.0f
#define ANGULAR_DAMPING 0.990f

typedef struct {
    float score;
    float perf;
    float episode_return;
    float episode_length;
    float success;
    float crash;
    float timeout;
    float start_altitude;
    float start_horizontal_speed;
    float terminal_fuel;
    float main_fuel_used;
    float side_fuel_used;
    float main_action_rate;
    float left_action_rate;
    float right_action_rate;
    float simultaneous_side_action_rate;
    float main_throttle_near_off_rate;
    float main_throttle_partial_rate;
    float main_throttle_near_full_rate;
    float attitude_command_near_limit_rate;
    float mean_raw_main_action;
    float mean_absolute_raw_main_action;
    float mean_absolute_raw_attitude_action;
    float main_throttle_altitude_0_100_sum;
    float absolute_attitude_altitude_0_100_sum;
    float action_steps_altitude_0_100;
    float main_throttle_altitude_100_500_sum;
    float absolute_attitude_altitude_100_500_sum;
    float action_steps_altitude_100_500;
    float main_throttle_altitude_500_plus_sum;
    float absolute_attitude_altitude_500_plus_sum;
    float action_steps_altitude_500_plus;
    float landed;
    float touchdown_vertical_speed;
    float touchdown_horizontal_speed;
    float touchdown_angle;
    float touchdown_angular_velocity;
    float failure_off_pad;
    float failure_vertical_speed;
    float failure_horizontal_speed;
    float failure_angle;
    float failure_angular_velocity;
    float failure_out_of_bounds;
    float failure_fuel_empty;
    float failure_unrecoverable_fuel;
    float episodes_100_500;
    float successes_100_500;
    float episodes_500_750;
    float successes_500_750;
    float episodes_750_1000;
    float successes_750_1000;
    float episodes_1000_1250;
    float successes_1000_1250;
    float episodes_1250_1500;
    float successes_1250_1500;
    float episodes_1500_plus;
    float successes_1500_plus;
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
    int rollout_horizon;
    int rollout_step;
    int terminal_display_ticks;
    int last_outcome;
    bool awaiting_rollout_reset;
    bool benchmark_single_episode;
    bool benchmark_complete;
    bool canonicalize_horizontal;
    bool horizontal_reflected;

    float altitude;
    float velocity;
    float x;
    float x_velocity;
    float start_x_velocity;
    float angle;
    float angular_velocity;
    float fuel;
    float start_altitude;
    float initial_altitude;
    float initial_downward_velocity;
    float reset_altitude_min;
    float reset_altitude_max;
    float reset_downward_velocity_min;
    float reset_downward_velocity_max;
    float reset_x_min;
    float reset_x_max;
    float reset_x_velocity_min;
    float reset_x_velocity_max;
    float reset_angle_min;
    float reset_angle_max;
    float reset_angular_velocity_min;
    float reset_angular_velocity_max;
    float dry_mass;
    float initial_fuel;
    float gravity;
    float thrust;
    float side_thrust;
    float fuel_burn_rate;
    float side_fuel_burn_rate;
    float moment_arm;
    float moment_inertia;
    float dt;
    float max_landing_speed;
    float terminal_reward_scale;
    float max_landing_x_speed;
    float max_landing_angle;
    float max_landing_angular_velocity;
    float world_width;
    float landing_pad_width;
    float max_velocity_obs;
    float max_x_velocity_obs;
    float max_angular_velocity_obs;
    float safety_reward_scale;
    float reward_gamma;
    float episode_return;
    float episode_main_fuel_used;
    float episode_side_fuel_used;
    float episode_main_action_steps;
    float episode_left_action_steps;
    float episode_right_action_steps;
    float episode_simultaneous_side_action_steps;
    float episode_main_throttle_near_off_steps;
    float episode_main_throttle_partial_steps;
    float episode_main_throttle_near_full_steps;
    float episode_attitude_command_near_limit_steps;
    float episode_raw_main_action_sum;
    float episode_absolute_raw_main_action_sum;
    float episode_absolute_raw_attitude_action_sum;
    float episode_main_throttle_altitude_0_100_sum;
    float episode_absolute_attitude_altitude_0_100_sum;
    float episode_action_steps_altitude_0_100;
    float episode_main_throttle_altitude_100_500_sum;
    float episode_absolute_attitude_altitude_100_500_sum;
    float episode_action_steps_altitude_100_500;
    float episode_main_throttle_altitude_500_plus_sum;
    float episode_absolute_attitude_altitude_500_plus_sum;
    float episode_action_steps_altitude_500_plus;
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
