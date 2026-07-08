#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"

#define OBS_SIZE 4

#define SCREEN_WIDTH 288
#define SCREEN_HEIGHT 512
#define RENDER_SCALE 1.75f

#define BIRD_WIDTH 34
#define BIRD_HEIGHT 24
#define BIRD_X 57

#define PIPE_WIDTH 52
#define PIPE_GAP 100
#define PIPE_SPACING 160
#define PIPE_SPEED 2.0f
#define NUM_PIPES 3

#define GRAVITY 0.45f
#define FLAP_VELOCITY -7.5f
#define MAX_FALL_SPEED 10.0f

#define GROUND_Y 400

#define NOOP 0
#define FLAP 1

typedef struct {
	float score;
	float episode_return;
	float episode_length;
	float n;
} Log;

typedef struct {
	float x; //fixed
	float y;
	float y_velocity;
	float height;
	float width;
} Bird;

typedef struct {
	float x;
	float gap_y;
	bool passed;
} Pipe;

typedef struct {
	Texture2D bird;
	Texture2D obstacle_top;
	Texture2D obstacle_bottom;
	Texture2D background;
	Texture2D ground;
} Client;

typedef struct {
	Log log;
	float* observations;
	float* actions;
	float* rewards;
	float* terminals;
	unsigned int rng;

	int num_agents;
	int width;
	int height;
	int ground_y;

	float gravity;
	float flap_velocity;
	float max_fall_speed;
	float pipe_speed;
	int pipe_width;
	int pipe_gap;
	int pipe_spacing;
	
	Client* client;
	Bird* bird;
	Pipe* pipes;
} Flappy;

Client* make_client(Flappy* env) {
	Client* client = (Client*)calloc(1, sizeof(Client));

	InitWindow(env->width * RENDER_SCALE, env->height * RENDER_SCALE, "PufferLib Flappy");
	SetTargetFPS(60);

	client->bird = LoadTexture("resources/flappy/bird.png");
	client->obstacle_top = LoadTexture("resources/flappy/obstacle_top.png");
	client->obstacle_bottom = LoadTexture("resources/flappy/obstacle_bottom.png");
	client->background = LoadTexture("resources/flappy/background.png");
	client->ground = LoadTexture("resources/flappy/ground.png");
	SetTextureFilter(client->bird, TEXTURE_FILTER_POINT);
	SetTextureFilter(client->obstacle_top, TEXTURE_FILTER_POINT);
	SetTextureFilter(client->obstacle_bottom, TEXTURE_FILTER_POINT);
	SetTextureFilter(client->background, TEXTURE_FILTER_POINT);
	SetTextureFilter(client->ground, TEXTURE_FILTER_POINT);

	return client;
}

void allocate(Flappy* env) {
	env->observations = (float* )calloc(OBS_SIZE, sizeof(float));
	env->actions = (float* )calloc(1, sizeof(float));
	env->rewards = (float* )calloc(1, sizeof(float));
	env->terminals = (float* )calloc(1, sizeof(float));
}


void c_init(Flappy* env){
	allocate(env);
	env->num_agents = 1;
	env->width = SCREEN_WIDTH;
	env->height = SCREEN_HEIGHT;
	env->ground_y = GROUND_Y;

	env->gravity = GRAVITY;
	env->flap_velocity = FLAP_VELOCITY;
	env->max_fall_speed = MAX_FALL_SPEED;

	env->pipe_speed = PIPE_SPEED;
	env->pipe_width = PIPE_WIDTH;
	env->pipe_gap = PIPE_GAP;
	env->pipe_spacing = PIPE_SPACING;
	env->client = NULL;

	env->bird = calloc(1, sizeof(Bird));
	env->bird->height = BIRD_HEIGHT;
	env->bird->width = BIRD_WIDTH;
	env->bird->x = BIRD_X;
	env->pipes = calloc(NUM_PIPES, sizeof(Pipe));
}

float random_gap_y(Flappy* env) {
	int margin = 50;
	int min_gap_y = margin;
	int max_gap_y = env->ground_y - env->pipe_gap - margin;
	if (max_gap_y <= min_gap_y) {
		return env->ground_y / 2.0f - env->pipe_gap / 2.0f;
	}

	return min_gap_y + rand_r(&env->rng) % (max_gap_y - min_gap_y + 1);
}

void c_reset(Flappy* env) {
	env->bird->y = env->ground_y / 2.0f;
	env->bird->y_velocity = 0.0f;

	for (int i = 0; i < NUM_PIPES; i++) {
		env->pipes[i].x = env->width + i * env->pipe_spacing;
		env->pipes[i].gap_y = random_gap_y(env);
		env->pipes[i].passed = false;
	}
}

Pipe* next_pipe(Flappy* env) {
	Pipe* best = &env->pipes[0];
	float best_x = env->width * 2.0f;
	float bird_x = env->bird->x;

	for (int i = 0; i < NUM_PIPES; i++) {
		float pipe_right = env->pipes[i].x + env->pipe_width;
		if (pipe_right >= bird_x && env->pipes[i].x < best_x) {
			best = &env->pipes[i];
			best_x = env->pipes[i].x;
		}
	}

	return best;
}

void compute_observations(Flappy* env) {
	Pipe* pipe = next_pipe(env);
	env->observations[0] = env->bird->y / env->ground_y;
	env->observations[1] = env->bird->y_velocity / env->max_fall_speed;
	env->observations[2] = (pipe->x + env->pipe_width - env->bird->x) / env->width;
	env->observations[3] = pipe->gap_y / env->ground_y;
}


void process_input(Flappy* env){
	int action = (int)env->actions[0];
	switch(action){
		case NOOP:
			break;
		case FLAP:
			env->bird->y_velocity = env->flap_velocity;
			break;
	}
}

void process_gravity(Flappy* env){
	env->bird->y_velocity += env->gravity;
	if (env->bird->y_velocity > env->max_fall_speed) {
		env->bird->y_velocity = env->max_fall_speed;
	}

	env->bird->y += env->bird->y_velocity;
	if(env->bird->y < 0){
		env->bird->y = 0;
		env->bird->y_velocity = 0;
	}
}


void move_pipes_and_process_collisions(Flappy* env){
	float bird_x_max = env->bird->x + env->bird->width;
	float bird_x_min = env->bird->x;
	float bird_y_min = env->bird->y;
	float bird_y_max = env->bird->y + env->bird->height;

	if (bird_y_min < 0 || bird_y_max > env->ground_y) {
		env->terminals[0] = 1.0f;
		env->rewards[0] = -1.0f;
		return;
	}

	for(int p = 0; p < NUM_PIPES; p++){
		Pipe* pipe = &env->pipes[p];
		pipe->x -= env->pipe_speed;

		float pipe_x_min = pipe->x;
		float pipe_x_max = pipe->x + env->pipe_width;
		bool overlaps_x = bird_x_max > pipe_x_min && bird_x_min < pipe_x_max;
		bool outside_gap = bird_y_min < pipe->gap_y ||
			bird_y_max > pipe->gap_y + env->pipe_gap;

		if (overlaps_x && outside_gap) {
			env->terminals[0] = 1.0f;
			env->rewards[0] = -1.0f;
			return;
		}

		if (!pipe->passed && pipe_x_max < bird_x_min) {
			pipe->passed = true;
			env->rewards[0] += 1.0f;
			env->log.score += 1.0f;
		}

		if (pipe_x_max < 0) {
			float rightmost_x = pipe->x;
			for (int i = 0; i < NUM_PIPES; i++) {
				if (env->pipes[i].x > rightmost_x) {
					rightmost_x = env->pipes[i].x;
				}
			}

			pipe->x = rightmost_x + env->pipe_spacing;
			pipe->gap_y = random_gap_y(env);
			pipe->passed = false;
		}
	}
}

void c_step(Flappy* env) {
	env->rewards[0] = 0.01f;
	env->terminals[0] = 0.0f;

	process_input(env);
	process_gravity(env);
	move_pipes_and_process_collisions(env);

	env->log.episode_return += env->rewards[0];
	env->log.episode_length += 1.0f;
	if (env->terminals[0]) {
		env->log.n += 1.0f;
		compute_observations(env);
		c_reset(env);
		return;
	}

	compute_observations(env);
}

void c_render(Flappy* env) {
	static int frame_idx = 0;

	if (env->client == NULL) {
		env->client = make_client(env);
	}

	if (IsKeyDown(KEY_ESCAPE)) {
		exit(0);
	}

	BeginDrawing();
	ClearBackground((Color){135, 206, 235, 255});
	BeginMode2D((Camera2D){
		.offset = (Vector2){0, 0},
		.target = (Vector2){0, 0},
		.rotation = 0,
		.zoom = RENDER_SCALE,
	});

	DrawTexturePro(
		env->client->background,
		(Rectangle){0, 0, env->client->background.width, env->client->background.height},
		(Rectangle){0, 0, env->width, env->ground_y},
		(Vector2){0, 0},
		0,
		WHITE
	);

	for (int i = 0; i < NUM_PIPES; i++) {
		Pipe* pipe = &env->pipes[i];
		float bottom_y = pipe->gap_y + env->pipe_gap;
		float bottom_height = env->ground_y - bottom_y;

		DrawTexturePro(
			env->client->obstacle_top,
			(Rectangle){1, 1, env->client->obstacle_top.width - 2, env->client->obstacle_top.height - 2},
			(Rectangle){pipe->x, 0, env->pipe_width, pipe->gap_y},
			(Vector2){0, 0},
			0,
			WHITE
		);
		DrawTexturePro(
			env->client->obstacle_bottom,
			(Rectangle){1, 1, env->client->obstacle_bottom.width - 2, env->client->obstacle_bottom.height - 2},
			(Rectangle){pipe->x, bottom_y, env->pipe_width, bottom_height},
			(Vector2){0, 0},
			0,
			WHITE
		);
	}

	DrawTexturePro(
		env->client->bird,
		(Rectangle){0, 0, env->client->bird.width, env->client->bird.height},
		(Rectangle){env->bird->x, env->bird->y, env->bird->width, env->bird->height},
		(Vector2){0, 0},
		0,
		WHITE
	);

	DrawTexturePro(
		env->client->ground,
		(Rectangle){0, 0, env->client->ground.width, env->client->ground.height},
		(Rectangle){0, env->ground_y, env->width, env->height - env->ground_y},
		(Vector2){0, 0},
		0,
		WHITE
	);

	EndMode2D();
	EndDrawing();

	char* frames_dir = getenv("FLAPPY_FRAMES_DIR");
	if (frames_dir != NULL) {
		int max_frames = atoi(getenv("FLAPPY_MAX_FRAMES"));
		if (max_frames <= 0 || frame_idx < max_frames) {
			TakeScreenshot(TextFormat("%s/flappy_%05d.png", frames_dir, frame_idx));
			frame_idx += 1;
		}
		if (max_frames > 0 && frame_idx >= max_frames) {
			exit(0);
		}
	}
}

void c_close(Flappy* env) {
	if (env->client != NULL) {
		if (env->client->bird.id) UnloadTexture(env->client->bird);
		if (env->client->obstacle_top.id) UnloadTexture(env->client->obstacle_top);
		if (env->client->obstacle_bottom.id) UnloadTexture(env->client->obstacle_bottom);
		if (env->client->background.id) UnloadTexture(env->client->background);
		if (env->client->ground.id) UnloadTexture(env->client->ground);
		free(env->client);
		env->client = NULL;
	}

	free(env->bird);
	free(env->pipes);

	if (IsWindowReady()) {
		CloseWindow();
	}
}
