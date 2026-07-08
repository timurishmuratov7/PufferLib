#include "booster_landing.h"
#include "env.c"

#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env BoosterLanding
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->max_episode_steps = dict_get(kwargs, "max_episode_steps")->value;
    env->initial_altitude = dict_get(kwargs, "initial_altitude")->value;
    env->initial_downward_velocity = dict_get(kwargs, "initial_downward_velocity")->value;
    env->reset_altitude_min = dict_get(kwargs, "reset_altitude_min")->value;
    env->reset_altitude_max = dict_get(kwargs, "reset_altitude_max")->value;
    env->reset_downward_velocity_min = dict_get(kwargs, "reset_downward_velocity_min")->value;
    env->reset_downward_velocity_max = dict_get(kwargs, "reset_downward_velocity_max")->value;
    env->dry_mass = dict_get(kwargs, "dry_mass")->value;
    env->initial_fuel = dict_get(kwargs, "initial_fuel")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->thrust = dict_get(kwargs, "thrust")->value;
    env->fuel_burn_rate = dict_get(kwargs, "fuel_burn_rate")->value;
    env->dt = dict_get(kwargs, "dt")->value;
    env->max_landing_speed = dict_get(kwargs, "max_landing_speed")->value;
    env->landing_pad_width = dict_get(kwargs, "landing_pad_width")->value;
    env->max_velocity_obs = dict_get(kwargs, "max_velocity_obs")->value;
    env->client = NULL;

    allocate(env);
    c_reset(env);
    compute_observations(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "crash", log->crash);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "n", log->n);
}
