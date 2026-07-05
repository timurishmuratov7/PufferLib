#include "flappy.h"

int main() {
	Flappy env = {0};
	c_init(&env);
	c_reset(&env);
	c_render(&env);

	while (!WindowShouldClose()) {
		env.actions[0] = NOOP;
		if (IsKeyPressed(KEY_SPACE)) {
			env.actions[0] = FLAP;
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
