#!/usr/bin/env python3
"""Evaluate one booster episode per native environment on fixed reset seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPO_ROOT / "benchmarks/booster_landing/upright_v1.json"
ALTITUDE_BINS = (
    "100_500",
    "500_750",
    "750_1000",
    "1000_1250",
    "1250_1500",
    "1500_plus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-seed booster benchmark with the native CUDA backend."
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--env-name")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--reset-seed", type=int)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--altitude-min", type=float)
    parser.add_argument("--altitude-max", type=float)
    parser.add_argument("--downward-velocity-min", type=float)
    parser.add_argument("--downward-velocity-max", type=float)
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--x-velocity-min", type=float)
    parser.add_argument("--x-velocity-max", type=float)
    parser.add_argument("--angular-velocity-min", type=float)
    parser.add_argument("--angular-velocity-max", type=float)
    parser.add_argument("--canonicalize-horizontal", type=int, choices=(0, 1))
    parser.add_argument("--max-landing-speed", type=float)
    parser.add_argument("--max-landing-x-speed", type=float)
    parser.add_argument("--max-landing-angle", type=float)
    parser.add_argument("--max-landing-angular-velocity", type=float)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_suite(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def suite_value(cli_value, evaluation: dict, key: str):
    return evaluation[key] if cli_value is None else cli_value


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_puffer_args(env_name: str = "booster_landing") -> dict:
    # load_config parses sys.argv. Hide this script's arguments while it reads
    # the regular booster config, then restore them for normal error reporting.
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        from pufferlib.pufferl import load_config

        return load_config(env_name)
    finally:
        sys.argv = argv


def wilson_interval(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    margin = z * math.sqrt(
        p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return [center - margin, center + margin]


def altitude_bin_results(metrics: dict[str, float], episodes: int) -> dict:
    results = {}
    counted = 0
    for name in ALTITUDE_BINS:
        count = int(round(metrics[f"episode_fraction_{name}"] * episodes))
        counted += count
        rate = metrics[f"score_{name}"]
        successes = int(round(rate * count))
        results[name] = {
            "episodes": count,
            "successes": successes,
            "rate": rate,
            "wilson_95": wilson_interval(successes, count) if count else None,
        }
    if counted != episodes:
        raise RuntimeError(f"altitude bins contain {counted} episodes, expected {episodes}")
    return results


def evaluate(
    checkpoint: Path,
    env_name: str,
    episodes: int,
    reset_seed: int,
    policy_seed: int,
    altitude_min: float,
    altitude_max: float,
    downward_velocity_min: float,
    downward_velocity_max: float,
    x_min: float | None,
    x_max: float | None,
    x_velocity_min: float,
    x_velocity_max: float,
    angular_velocity_min: float,
    angular_velocity_max: float,
    canonicalize_horizontal: bool,
    max_landing_speed: float | None,
    max_landing_x_speed: float,
    max_landing_angle: float | None,
    max_landing_angular_velocity: float | None,
    gpu_id: int,
    num_buffers: int,
    num_threads: int,
    quiet: bool,
) -> tuple[dict, dict]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if episodes % num_buffers != 0:
        raise ValueError("episodes must be divisible by num_buffers")
    if altitude_max < altitude_min:
        raise ValueError("altitude_max must be greater than or equal to altitude_min")
    if downward_velocity_min < 0.0:
        raise ValueError("downward_velocity_min must be nonnegative")
    if downward_velocity_max < downward_velocity_min:
        raise ValueError(
            "downward_velocity_max must be greater than or equal to downward_velocity_min"
        )
    if x_min is not None and x_max is not None and x_max < x_min:
        raise ValueError("x_max must be greater than or equal to x_min")
    if x_velocity_max < x_velocity_min:
        raise ValueError("x_velocity_max must be greater than or equal to x_velocity_min")
    if angular_velocity_max < angular_velocity_min:
        raise ValueError(
            "angular_velocity_max must be greater than or equal to angular_velocity_min"
        )
    if max_landing_x_speed < 0.0:
        raise ValueError("max_landing_x_speed must be nonnegative")

    args = load_puffer_args(env_name)
    rollout_horizon = int(args["env"]["rollout_horizon"])
    args["env"]["benchmark_single_episode"] = 1
    args["env"]["reset_seed"] = reset_seed
    args["env"]["reset_altitude_min"] = altitude_min
    args["env"]["reset_altitude_max"] = altitude_max
    args["env"]["reset_downward_velocity_min"] = downward_velocity_min
    args["env"]["reset_downward_velocity_max"] = downward_velocity_max
    if x_min is not None:
        args["env"]["reset_x_min"] = x_min
    if x_max is not None:
        args["env"]["reset_x_max"] = x_max
    args["env"]["reset_x_velocity_min"] = x_velocity_min
    args["env"]["reset_x_velocity_max"] = x_velocity_max
    args["env"]["reset_angular_velocity_min"] = angular_velocity_min
    args["env"]["reset_angular_velocity_max"] = angular_velocity_max
    args["env"]["canonicalize_horizontal"] = int(canonicalize_horizontal)
    if max_landing_speed is not None:
        args["env"]["max_landing_speed"] = max_landing_speed
    args["env"]["max_landing_x_speed"] = max_landing_x_speed
    if max_landing_angle is not None:
        args["env"]["max_landing_angle"] = max_landing_angle
    if max_landing_angular_velocity is not None:
        args["env"]["max_landing_angular_velocity"] = max_landing_angular_velocity
    args["vec"]["total_agents"] = episodes
    args["vec"]["num_buffers"] = num_buffers
    args["vec"]["num_threads"] = num_threads
    args["train"]["horizon"] = rollout_horizon
    args["train"]["minibatch_size"] = episodes
    args["train"]["total_timesteps"] = episodes * rollout_horizon
    args["seed"] = policy_seed
    args["train"]["seed"] = policy_seed
    args["gpu_id"] = gpu_id
    args["rank"] = 0
    args["world_size"] = 1
    args["nccl_id"] = "None"
    args["reset_state"] = True
    args["profile"] = False

    from pufferlib import _C

    runner = None
    metrics: dict[str, float] = {}
    try:
        runner = _C.create_pufferl(args)
        _C.load_weights(runner, str(checkpoint))

        max_rollouts = math.ceil(args["env"]["max_episode_steps"] / rollout_horizon) + 2
        for rollout in range(1, max_rollouts + 1):
            _C.rollouts(runner)
            raw = _C.eval_log(runner)
            metrics = {key: float(value) for key, value in dict(raw.get("env", {})).items()}
            completed = int(round(metrics.get("n", 0.0)))
            if not quiet:
                print(
                    f"rollout {rollout}/{max_rollouts}: {completed}/{episodes} episodes",
                    file=sys.stderr,
                )
            if completed == episodes:
                break
            if completed > episodes:
                raise RuntimeError(
                    f"benchmark recorded {completed} episodes for {episodes} environments"
                )
        else:
            completed = int(round(metrics.get("n", 0.0)))
            raise RuntimeError(
                f"only {completed}/{episodes} environments completed after {max_rollouts} rollouts"
            )
    finally:
        if runner is not None:
            _C.close(runner)

    return metrics, args


def main() -> None:
    cli = parse_args()
    suite_path = cli.suite.resolve()
    suite = load_suite(suite_path)
    evaluation = suite["evaluation"]
    env_name = cli.env_name or suite.get("env_name", "booster_landing")

    checkpoint_arg = cli.checkpoint or Path(suite["checkpoint"]["path"])
    checkpoint = resolve_repo_path(checkpoint_arg).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if cli.checkpoint is None:
        expected_hash = suite["checkpoint"]["sha256"]
        if checkpoint_hash != expected_hash:
            raise RuntimeError(
                "frozen checkpoint hash mismatch: "
                f"expected {expected_hash}, got {checkpoint_hash}"
            )

    episodes = int(suite_value(cli.episodes, evaluation, "episodes"))
    reset_seed = int(suite_value(cli.reset_seed, evaluation, "reset_seed"))
    policy_seed = int(suite_value(cli.policy_seed, evaluation, "policy_seed"))
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
    x_min = evaluation.get("x_min") if cli.x_min is None else cli.x_min
    x_max = evaluation.get("x_max") if cli.x_max is None else cli.x_max
    angular_velocity_min = float(
        evaluation.get("angular_velocity_min", 0.0)
        if cli.angular_velocity_min is None
        else cli.angular_velocity_min
    )
    angular_velocity_max = float(
        evaluation.get("angular_velocity_max", 0.0)
        if cli.angular_velocity_max is None
        else cli.angular_velocity_max
    )
    canonicalize_horizontal = bool(
        evaluation.get("canonicalize_horizontal", False)
        if cli.canonicalize_horizontal is None
        else cli.canonicalize_horizontal
    )
    max_landing_speed = (
        evaluation.get("max_landing_speed")
        if cli.max_landing_speed is None
        else cli.max_landing_speed
    )
    max_landing_x_speed = float(
        suite_value(cli.max_landing_x_speed, evaluation, "max_landing_x_speed")
    )
    max_landing_angle = (
        evaluation.get("max_landing_angle")
        if cli.max_landing_angle is None
        else cli.max_landing_angle
    )
    max_landing_angular_velocity = (
        evaluation.get("max_landing_angular_velocity")
        if cli.max_landing_angular_velocity is None
        else cli.max_landing_angular_velocity
    )

    metrics, puffer_args = evaluate(
        checkpoint=checkpoint,
        env_name=env_name,
        episodes=episodes,
        reset_seed=reset_seed,
        policy_seed=policy_seed,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        downward_velocity_min=downward_velocity_min,
        downward_velocity_max=downward_velocity_max,
        x_min=x_min,
        x_max=x_max,
        x_velocity_min=x_velocity_min,
        x_velocity_max=x_velocity_max,
        angular_velocity_min=angular_velocity_min,
        angular_velocity_max=angular_velocity_max,
        canonicalize_horizontal=canonicalize_horizontal,
        max_landing_speed=max_landing_speed,
        max_landing_x_speed=max_landing_x_speed,
        max_landing_angle=max_landing_angle,
        max_landing_angular_velocity=max_landing_angular_velocity,
        gpu_id=cli.gpu_id,
        num_buffers=cli.num_buffers,
        num_threads=cli.num_threads,
        quiet=cli.quiet,
    )

    successes = int(round(metrics["score"] * episodes))
    result = {
        "schema_version": 1,
        "suite": suite["id"],
        "env_name": env_name,
        "checkpoint": {
            "path": str(checkpoint_arg),
            "sha256": checkpoint_hash,
        },
        "evaluation": {
            "episodes": episodes,
            "reset_seed": reset_seed,
            "policy_seed": policy_seed,
            "altitude_min": altitude_min,
            "altitude_max": altitude_max,
            "downward_velocity_min": downward_velocity_min,
            "downward_velocity_max": downward_velocity_max,
            "x_min": puffer_args["env"]["reset_x_min"],
            "x_max": puffer_args["env"]["reset_x_max"],
            "x_velocity_min": x_velocity_min,
            "x_velocity_max": x_velocity_max,
            "angular_velocity_min": angular_velocity_min,
            "angular_velocity_max": angular_velocity_max,
            "canonicalize_horizontal": canonicalize_horizontal,
            "max_landing_speed": puffer_args["env"]["max_landing_speed"],
            "max_landing_x_speed": puffer_args["env"]["max_landing_x_speed"],
            "max_landing_angle": puffer_args["env"]["max_landing_angle"],
            "max_landing_angular_velocity": puffer_args["env"]["max_landing_angular_velocity"],
            "rollout_horizon": puffer_args["train"]["horizon"],
            "num_buffers": cli.num_buffers,
            "num_threads": cli.num_threads,
        },
        "score": {
            "successes": successes,
            "rate": metrics["score"],
            "wilson_95": wilson_interval(successes, episodes),
        },
        "altitude_bins": altitude_bin_results(metrics, episodes),
        "metrics": metrics,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if cli.output is not None:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
