#include "booster_landing.h"
#include "env.c"

#define NUM_ATNS 3
#define ACT_SIZES {2, 2, 2}
#define OBS_TENSOR_T FloatTensor

#define Env BoosterLanding
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->max_episode_steps = dict_get(kwargs, "max_episode_steps")->value;
    env->rollout_horizon = dict_get(kwargs, "rollout_horizon")->value;
    env->rollout_step = 0;
    env->initial_altitude = dict_get(kwargs, "initial_altitude")->value;
    env->initial_downward_velocity = dict_get(kwargs, "initial_downward_velocity")->value;
    env->reset_altitude_min = dict_get(kwargs, "reset_altitude_min")->value;
    env->reset_altitude_max = dict_get(kwargs, "reset_altitude_max")->value;
    env->reset_downward_velocity_min = dict_get(kwargs, "reset_downward_velocity_min")->value;
    env->reset_downward_velocity_max = dict_get(kwargs, "reset_downward_velocity_max")->value;
    env->reset_x_min = dict_get(kwargs, "reset_x_min")->value;
    env->reset_x_max = dict_get(kwargs, "reset_x_max")->value;
    env->reset_angle_min = dict_get(kwargs, "reset_angle_min")->value;
    env->reset_angle_max = dict_get(kwargs, "reset_angle_max")->value;
    env->dry_mass = dict_get(kwargs, "dry_mass")->value;
    env->initial_fuel = dict_get(kwargs, "initial_fuel")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->thrust = dict_get(kwargs, "thrust")->value;
    env->side_thrust = dict_get(kwargs, "side_thrust")->value;
    env->fuel_burn_rate = dict_get(kwargs, "fuel_burn_rate")->value;
    env->side_fuel_burn_rate = dict_get(kwargs, "side_fuel_burn_rate")->value;
    env->moment_arm = dict_get(kwargs, "moment_arm")->value;
    env->moment_inertia = dict_get(kwargs, "moment_inertia")->value;
    env->dt = dict_get(kwargs, "dt")->value;
    env->max_landing_speed = dict_get(kwargs, "max_landing_speed")->value;
    env->terminal_reward_scale = dict_get(kwargs, "terminal_reward_scale")->value;
    env->max_landing_x_speed = dict_get(kwargs, "max_landing_x_speed")->value;
    env->max_landing_angle = dict_get(kwargs, "max_landing_angle")->value;
    env->max_landing_angular_velocity = dict_get(kwargs, "max_landing_angular_velocity")->value;
    env->world_width = dict_get(kwargs, "world_width")->value;
    env->landing_pad_width = dict_get(kwargs, "landing_pad_width")->value;
    env->max_velocity_obs = dict_get(kwargs, "max_velocity_obs")->value;
    env->max_x_velocity_obs = dict_get(kwargs, "max_x_velocity_obs")->value;
    env->max_angular_velocity_obs = dict_get(kwargs, "max_angular_velocity_obs")->value;
    env->safety_reward_scale = dict_get(kwargs, "safety_reward_scale")->value;
    env->reward_gamma = dict_get(kwargs, "reward_gamma")->value;
    env->client = NULL;

    allocate(env);
    c_reset(env);
    compute_observations(env);
}

void my_log(Log* log, Dict* out) {
    float landed = log->landed;

    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "crash", log->crash);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "start_altitude", log->start_altitude);
    dict_set(out, "terminal_fuel", log->terminal_fuel);
    dict_set(out, "main_fuel_used", log->main_fuel_used);
    dict_set(out, "side_fuel_used", log->side_fuel_used);
    dict_set(out, "main_action_rate", log->main_action_rate);
    dict_set(out, "left_action_rate", log->left_action_rate);
    dict_set(out, "right_action_rate", log->right_action_rate);
    dict_set(out, "simultaneous_side_action_rate", log->simultaneous_side_action_rate);
    dict_set(out, "touchdown_vertical_speed",
        landed > 0.0f ? log->touchdown_vertical_speed / landed : 0.0f);
    dict_set(out, "touchdown_horizontal_speed",
        landed > 0.0f ? log->touchdown_horizontal_speed / landed : 0.0f);
    dict_set(out, "touchdown_angle",
        landed > 0.0f ? log->touchdown_angle / landed : 0.0f);
    dict_set(out, "touchdown_angular_velocity",
        landed > 0.0f ? log->touchdown_angular_velocity / landed : 0.0f);
    dict_set(out, "failure_off_pad", log->failure_off_pad);
    dict_set(out, "failure_vertical_speed", log->failure_vertical_speed);
    dict_set(out, "failure_horizontal_speed", log->failure_horizontal_speed);
    dict_set(out, "failure_angle", log->failure_angle);
    dict_set(out, "failure_angular_velocity", log->failure_angular_velocity);
    dict_set(out, "failure_out_of_bounds", log->failure_out_of_bounds);
    dict_set(out, "failure_fuel_empty", log->failure_fuel_empty);
    dict_set(out, "failure_unrecoverable_fuel", log->failure_unrecoverable_fuel);
    dict_set(out, "score_100_500", log->episodes_100_500 > 0.0f ?
        log->successes_100_500 / log->episodes_100_500 : 0.0f);
    dict_set(out, "score_500_750", log->episodes_500_750 > 0.0f ?
        log->successes_500_750 / log->episodes_500_750 : 0.0f);
    dict_set(out, "score_750_1000", log->episodes_750_1000 > 0.0f ?
        log->successes_750_1000 / log->episodes_750_1000 : 0.0f);
    dict_set(out, "score_1000_1250", log->episodes_1000_1250 > 0.0f ?
        log->successes_1000_1250 / log->episodes_1000_1250 : 0.0f);
    dict_set(out, "score_1250_1500", log->episodes_1250_1500 > 0.0f ?
        log->successes_1250_1500 / log->episodes_1250_1500 : 0.0f);
    dict_set(out, "score_1500_plus", log->episodes_1500_plus > 0.0f ?
        log->successes_1500_plus / log->episodes_1500_plus : 0.0f);
    dict_set(out, "n", log->n);
}
