#!/usr/bin/env python3
"""Evaluate a deterministic guidance controller on the native booster environment."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import sys

import numpy as np

from eval_booster_landing import (
    ALTITUDE_BINS,
    DEFAULT_SUITE,
    load_puffer_args,
    load_suite,
    suite_value,
    wilson_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_SUITE = (
    REPO_ROOT / "benchmarks/booster_landing/angular_velocity_005_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic booster guidance on fixed native resets."
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_CONTROLLER_SUITE)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--reset-seed", type=int)
    parser.add_argument("--altitude-min", type=float)
    parser.add_argument("--altitude-max", type=float)
    parser.add_argument("--downward-velocity-min", type=float)
    parser.add_argument("--downward-velocity-max", type=float)
    parser.add_argument("--x-velocity-min", type=float)
    parser.add_argument("--x-velocity-max", type=float)
    parser.add_argument("--angular-velocity-min", type=float)
    parser.add_argument("--angular-velocity-max", type=float)
    parser.add_argument("--canonicalize-horizontal", type=int, choices=(0, 1))
    parser.add_argument("--max-landing-x-speed", type=float)
    parser.add_argument("--target-touchdown-speed", type=float, default=2.0)
    parser.add_argument("--landing-margin", type=float, default=3.0)
    parser.add_argument("--effective-fuel-fraction", type=float, default=1.0)
    parser.add_argument("--x-gain", type=float, default=0.001)
    parser.add_argument("--x-velocity-gain", type=float, default=0.07)
    parser.add_argument("--max-guidance-angle", type=float, default=0.25)
    parser.add_argument("--upright-altitude", type=float, default=40.0)
    parser.add_argument("--attitude-gain", type=float, default=4.0)
    parser.add_argument("--angular-velocity-gain", type=float, default=2.5)
    parser.add_argument("--torque-deadband", type=float, default=0.003)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def pointer_array(pointer: int, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    raw = (ctypes.c_float * count).from_address(pointer)
    return np.ctypeslib.as_array(raw).reshape(shape)


def optional_suite_value(cli_value, evaluation: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return evaluation.get(key, default)


def altitude_bin_name(altitude: np.ndarray) -> np.ndarray:
    names = np.full(altitude.shape, "1500_plus", dtype=object)
    names[altitude < 1500.0] = "1250_1500"
    names[altitude < 1250.0] = "1000_1250"
    names[altitude < 1000.0] = "750_1000"
    names[altitude < 750.0] = "500_750"
    names[altitude < 500.0] = "100_500"
    return names


def run_controller(cli: argparse.Namespace) -> dict:
    suite_path = cli.suite.resolve()
    suite = load_suite(suite_path)
    evaluation = suite["evaluation"]

    episodes = int(suite_value(cli.episodes, evaluation, "episodes"))
    reset_seed = int(suite_value(cli.reset_seed, evaluation, "reset_seed"))
    altitude_min = float(suite_value(cli.altitude_min, evaluation, "altitude_min"))
    altitude_max = float(suite_value(cli.altitude_max, evaluation, "altitude_max"))
    downward_velocity_min = float(
        suite_value(cli.downward_velocity_min, evaluation, "downward_velocity_min")
    )
    downward_velocity_max = float(
        suite_value(cli.downward_velocity_max, evaluation, "downward_velocity_max")
    )
    x_velocity_min = float(suite_value(cli.x_velocity_min, evaluation, "x_velocity_min"))
    x_velocity_max = float(suite_value(cli.x_velocity_max, evaluation, "x_velocity_max"))
    angular_velocity_min = float(
        optional_suite_value(
            cli.angular_velocity_min, evaluation, "angular_velocity_min", 0.0
        )
    )
    angular_velocity_max = float(
        optional_suite_value(
            cli.angular_velocity_max, evaluation, "angular_velocity_max", 0.0
        )
    )
    canonicalize_horizontal = bool(
        optional_suite_value(
            cli.canonicalize_horizontal,
            evaluation,
            "canonicalize_horizontal",
            False,
        )
    )
    max_landing_x_speed = float(
        suite_value(cli.max_landing_x_speed, evaluation, "max_landing_x_speed")
    )

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if altitude_max < altitude_min:
        raise ValueError("altitude_max must be greater than or equal to altitude_min")
    if downward_velocity_max < downward_velocity_min:
        raise ValueError(
            "downward_velocity_max must be greater than or equal to downward_velocity_min"
        )
    if x_velocity_max < x_velocity_min:
        raise ValueError("x_velocity_max must be greater than or equal to x_velocity_min")
    if angular_velocity_max < angular_velocity_min:
        raise ValueError(
            "angular_velocity_max must be greater than or equal to angular_velocity_min"
        )

    args = load_puffer_args()
    env_args = args["env"]
    env_args["benchmark_single_episode"] = 1
    env_args["reset_seed"] = reset_seed
    env_args["reset_altitude_min"] = altitude_min
    env_args["reset_altitude_max"] = altitude_max
    env_args["reset_downward_velocity_min"] = downward_velocity_min
    env_args["reset_downward_velocity_max"] = downward_velocity_max
    env_args["reset_x_velocity_min"] = x_velocity_min
    env_args["reset_x_velocity_max"] = x_velocity_max
    env_args["reset_angular_velocity_min"] = angular_velocity_min
    env_args["reset_angular_velocity_max"] = angular_velocity_max
    env_args["canonicalize_horizontal"] = int(canonicalize_horizontal)
    env_args["max_landing_x_speed"] = max_landing_x_speed
    args["vec"]["total_agents"] = episodes
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1

    from pufferlib import _C

    vec = _C.create_vec(args, 0)
    try:
        observations = pointer_array(vec.obs_ptr, (episodes, vec.obs_size))
        rewards = pointer_array(vec.rewards_ptr, (episodes,))
        terminals = pointer_array(vec.terminals_ptr, (episodes,))
        actions = np.zeros((episodes, vec.num_atns), dtype=np.float32)

        vec.reset()
        initial_observations = observations.copy()
        active = np.ones(episodes, dtype=bool)
        ticks = np.zeros(episodes, dtype=np.int32)
        episode_returns = np.zeros(episodes, dtype=np.float64)
        torque_accumulator = np.zeros(episodes, dtype=np.float64)
        main_steps = np.zeros(episodes, dtype=np.float64)
        left_steps = np.zeros(episodes, dtype=np.float64)
        right_steps = np.zeros(episodes, dtype=np.float64)
        main_fuel_used = np.zeros(episodes, dtype=np.float64)
        side_fuel_used = np.zeros(episodes, dtype=np.float64)
        terminal_observations = np.zeros_like(initial_observations)

        initial_altitude_scale = float(env_args["initial_altitude"])
        max_velocity_obs = float(env_args["max_velocity_obs"])
        world_width = float(env_args["world_width"])
        max_x_velocity_obs = float(env_args["max_x_velocity_obs"])
        max_angular_velocity_obs = float(env_args["max_angular_velocity_obs"])
        initial_fuel = float(env_args["initial_fuel"])
        dry_mass = float(env_args["dry_mass"])
        gravity = float(env_args["gravity"])
        thrust = float(env_args["thrust"])
        side_thrust = float(env_args["side_thrust"])
        moment_arm = float(env_args["moment_arm"])
        moment_inertia = float(env_args["moment_inertia"])
        fuel_burn_rate = float(env_args["fuel_burn_rate"])
        side_fuel_burn_rate = float(env_args["side_fuel_burn_rate"])
        dt = float(env_args["dt"])
        max_steps = int(env_args["max_episode_steps"])

        for _ in range(max_steps + 1):
            if not np.any(active):
                break

            altitude = observations[:, 0] * initial_altitude_scale
            vertical_velocity = observations[:, 1] * max_velocity_obs
            x = observations[:, 2] * world_width
            x_velocity = observations[:, 3] * max_x_velocity_obs
            angle = observations[:, 4] * math.pi
            angular_velocity = observations[:, 5] * max_angular_velocity_obs
            fuel = observations[:, 6] * initial_fuel

            estimated_mass = dry_mass + cli.effective_fuel_fraction * fuel
            vertical_thrust = thrust * np.maximum(np.cos(angle), 0.0)
            braking_acceleration = np.maximum(
                vertical_thrust / np.maximum(estimated_mass, 1.0) - gravity,
                0.001,
            )
            usable_altitude = np.maximum(altitude - cli.landing_margin, 0.0)
            switch_speed = np.sqrt(
                cli.target_touchdown_speed**2
                + 2.0 * braking_acceleration * usable_altitude
            )
            downward_speed = np.maximum(-vertical_velocity, 0.0)
            main_command = downward_speed >= switch_speed
            main_command |= (
                (altitude <= cli.landing_margin)
                & (downward_speed > cli.target_touchdown_speed)
            )

            angle_command = np.clip(
                -cli.x_gain * x - cli.x_velocity_gain * x_velocity,
                -cli.max_guidance_angle,
                cli.max_guidance_angle,
            )
            upright_scale = np.clip(
                altitude / max(cli.upright_altitude, 0.001), 0.0, 1.0
            )
            angle_command *= upright_scale
            angular_acceleration_command = (
                cli.attitude_gain * (angle_command - angle)
                - cli.angular_velocity_gain * angular_velocity
            )
            angular_acceleration_command[
                np.abs(angular_acceleration_command) < cli.torque_deadband
            ] = 0.0

            side_angular_acceleration = side_thrust * moment_arm / moment_inertia
            normalized_torque = np.clip(
                angular_acceleration_command / side_angular_acceleration,
                -1.0,
                1.0,
            )
            torque_accumulator[active] += normalized_torque[active]
            left_command = torque_accumulator >= 1.0
            right_command = torque_accumulator <= -1.0
            torque_accumulator[left_command] -= 1.0
            torque_accumulator[right_command] += 1.0

            actions.fill(0.0)
            actions[active, 0] = main_command[active]
            actions[active, 1] = left_command[active]
            actions[active, 2] = right_command[active]

            fuel_before = fuel.copy()
            active_before = active.copy()
            main_steps[active_before] += main_command[active_before]
            left_steps[active_before] += left_command[active_before]
            right_steps[active_before] += right_command[active_before]

            vec.cpu_step(actions.ctypes.data)
            ticks[active_before] += 1
            episode_returns[active_before] += rewards[active_before]

            fuel_after = observations[:, 6] * initial_fuel
            fuel_drop = np.maximum(fuel_before - fuel_after, 0.0)
            requested_main = main_command * fuel_burn_rate * dt
            requested_side = (left_command + right_command) * side_fuel_burn_rate * dt
            requested_total = requested_main + requested_side
            split = np.divide(
                fuel_drop,
                requested_total,
                out=np.zeros_like(fuel_drop),
                where=requested_total > 0.0,
            )
            main_fuel_used[active_before] += requested_main[active_before] * split[active_before]
            side_fuel_used[active_before] += requested_side[active_before] * split[active_before]

            completed = active_before & (terminals > 0.5)
            terminal_observations[completed] = observations[completed]
            active[completed] = False
        else:
            raise RuntimeError("controller benchmark exceeded the configured episode horizon")

        if np.any(active):
            raise RuntimeError(f"{np.count_nonzero(active)} controller episodes did not complete")
    finally:
        vec.close()

    start_altitude = initial_observations[:, 0] * initial_altitude_scale
    start_x_velocity = np.abs(initial_observations[:, 3] * max_x_velocity_obs)
    start_angular_velocity = np.abs(
        initial_observations[:, 5] * max_angular_velocity_obs
    )
    terminal_altitude = terminal_observations[:, 0] * initial_altitude_scale
    terminal_vertical_velocity = terminal_observations[:, 1] * max_velocity_obs
    terminal_x = terminal_observations[:, 2] * world_width
    terminal_x_velocity = terminal_observations[:, 3] * max_x_velocity_obs
    terminal_angle = terminal_observations[:, 4] * math.pi
    terminal_angular_velocity = (
        terminal_observations[:, 5] * max_angular_velocity_obs
    )
    terminal_fuel = terminal_observations[:, 6] * initial_fuel

    landed = terminal_altitude <= 1e-4
    on_pad = np.abs(terminal_x) <= float(env_args["landing_pad_width"]) / 2.0
    stable_vertical = (
        np.abs(terminal_vertical_velocity) <= float(env_args["max_landing_speed"])
    )
    stable_horizontal = np.abs(terminal_x_velocity) <= max_landing_x_speed
    stable_angle = np.abs(terminal_angle) <= float(env_args["max_landing_angle"])
    stable_angular_velocity = np.abs(terminal_angular_velocity) <= float(
        env_args["max_landing_angular_velocity"]
    )
    success = (
        landed
        & on_pad
        & stable_vertical
        & stable_horizontal
        & stable_angle
        & stable_angular_velocity
    )
    fuel_empty = terminal_fuel <= 0.001
    out_of_bounds = np.abs(terminal_x) >= world_width * (1.0 - 1e-6)
    timeout = (~landed) & (~fuel_empty) & (~out_of_bounds) & (ticks >= max_steps)
    crash = (~success) & (~timeout)

    successes = int(np.count_nonzero(success))
    landed_count = int(np.count_nonzero(landed))
    episode_steps = np.maximum(ticks, 1)
    bin_names = altitude_bin_name(start_altitude)
    altitude_bins = {}
    for name in ALTITUDE_BINS:
        mask = bin_names == name
        count = int(np.count_nonzero(mask))
        bin_successes = int(np.count_nonzero(success & mask))
        altitude_bins[name] = {
            "episodes": count,
            "successes": bin_successes,
            "rate": bin_successes / count if count else 0.0,
            "wilson_95": wilson_interval(bin_successes, count) if count else None,
        }

    def rate(mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask) / episodes)

    metrics = {
        "n": float(episodes),
        "score": successes / episodes,
        "success": successes / episodes,
        "crash": rate(crash),
        "timeout": rate(timeout),
        "episode_return": float(np.mean(episode_returns)),
        "episode_length": float(np.mean(ticks)),
        "start_altitude": float(np.mean(start_altitude)),
        "start_horizontal_speed": float(np.mean(start_x_velocity)),
        "start_angular_velocity": float(np.mean(start_angular_velocity)),
        "terminal_fuel": float(np.mean(terminal_fuel)),
        "main_fuel_used": float(np.mean(main_fuel_used)),
        "side_fuel_used": float(np.mean(side_fuel_used)),
        "main_action_rate": float(np.mean(main_steps / episode_steps)),
        "left_action_rate": float(np.mean(left_steps / episode_steps)),
        "right_action_rate": float(np.mean(right_steps / episode_steps)),
        "simultaneous_side_action_rate": 0.0,
        "touchdown_vertical_speed": float(
            np.mean(np.abs(terminal_vertical_velocity[landed]))
        ) if landed_count else 0.0,
        "touchdown_horizontal_speed": float(
            np.mean(np.abs(terminal_x_velocity[landed]))
        ) if landed_count else 0.0,
        "touchdown_angle": float(np.mean(np.abs(terminal_angle[landed])))
        if landed_count
        else 0.0,
        "touchdown_angular_velocity": float(
            np.mean(np.abs(terminal_angular_velocity[landed]))
        ) if landed_count else 0.0,
        "failure_off_pad": rate(landed & (~success) & (~on_pad)),
        "failure_vertical_speed": rate(landed & (~success) & (~stable_vertical)),
        "failure_horizontal_speed": rate(
            landed & (~success) & (~stable_horizontal)
        ),
        "failure_angle": rate(landed & (~success) & (~stable_angle)),
        "failure_angular_velocity": rate(
            landed & (~success) & (~stable_angular_velocity)
        ),
        "failure_out_of_bounds": rate((~success) & out_of_bounds),
        "failure_fuel_empty": rate((~success) & fuel_empty),
        "failure_unrecoverable_fuel": rate(
            (~success) & (~landed) & fuel_empty
        ),
    }

    controller = {
        "type": "stopping_distance_attitude_pd",
        "target_touchdown_speed": cli.target_touchdown_speed,
        "landing_margin": cli.landing_margin,
        "effective_fuel_fraction": cli.effective_fuel_fraction,
        "x_gain": cli.x_gain,
        "x_velocity_gain": cli.x_velocity_gain,
        "max_guidance_angle": cli.max_guidance_angle,
        "upright_altitude": cli.upright_altitude,
        "attitude_gain": cli.attitude_gain,
        "angular_velocity_gain": cli.angular_velocity_gain,
        "torque_deadband": cli.torque_deadband,
    }
    return {
        "schema_version": 1,
        "suite": suite["id"],
        "controller": controller,
        "evaluation": {
            "episodes": episodes,
            "reset_seed": reset_seed,
            "altitude_min": altitude_min,
            "altitude_max": altitude_max,
            "downward_velocity_min": downward_velocity_min,
            "downward_velocity_max": downward_velocity_max,
            "x_velocity_min": x_velocity_min,
            "x_velocity_max": x_velocity_max,
            "angular_velocity_min": angular_velocity_min,
            "angular_velocity_max": angular_velocity_max,
            "canonicalize_horizontal": canonicalize_horizontal,
            "max_landing_x_speed": max_landing_x_speed,
        },
        "score": {
            "successes": successes,
            "rate": successes / episodes,
            "wilson_95": wilson_interval(successes, episodes),
        },
        "altitude_bins": altitude_bins,
        "metrics": metrics,
    }


def main() -> None:
    cli = parse_args()
    result = run_controller(cli)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if cli.output is not None:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(rendered, encoding="utf-8")
    if not cli.quiet:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
