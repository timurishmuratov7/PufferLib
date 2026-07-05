#include "flappy.h"

#define OBS_SIZE 4
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env Flappy
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
	env->num_agents = 1;
	env->width = dict_get(kwargs, "width")->value;
	env->height = dict_get(kwargs, "height")->value;
	env->ground_y = dict_get(kwargs, "ground_y")->value;
	env->gravity = dict_get(kwargs, "gravity")->value;
	env->flap_velocity = dict_get(kwargs, "flap_velocity")->value;
	env->max_fall_speed = dict_get(kwargs, "max_fall_speed")->value;
	env->pipe_speed = dict_get(kwargs, "pipe_speed")->value;
	env->pipe_width = dict_get(kwargs, "pipe_width")->value;
	env->pipe_gap = dict_get(kwargs, "pipe_gap")->value;
	env->pipe_spacing = dict_get(kwargs, "pipe_spacing")->value;
	env->client = NULL;

	allocate(env);
	env->bird = calloc(1, sizeof(Bird));
	env->bird->height = BIRD_HEIGHT;
	env->bird->width = BIRD_WIDTH;
	env->bird->x = BIRD_X;
	env->pipes = calloc(NUM_PIPES, sizeof(Pipe));
	c_reset(env);
	compute_observations(env);
}

void my_log(Log* log, Dict* out) {
	dict_set(out, "score", log->score);
	dict_set(out, "episode_return", log->episode_return);
	dict_set(out, "episode_length", log->episode_length);
	dict_set(out, "n", log->n);
}
