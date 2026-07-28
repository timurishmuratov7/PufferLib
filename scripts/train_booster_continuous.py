#!/usr/bin/env python3
"""Train the continuous booster while deferring broken native cleanup."""

from __future__ import annotations

import os
import sys
import traceback


ENV_NAME = "booster_landing_continuous"


def exit_without_native_cleanup(status: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def main() -> None:
    from pufferlib import _C
    from pufferlib import pufferl

    compiled_env = getattr(_C, "env_name", None)
    if compiled_env != ENV_NAME:
        raise RuntimeError(
            f"build.sh was run for {compiled_env}, not {ENV_NAME}"
        )

    # Native continuous training currently crashes in _C.close after the final
    # checkpoint is saved. Keep the runner alive until process exit so PufferLib
    # can serialize its normal JSON log first; the OS then reclaims GPU memory.
    deferred_runners = []

    def defer_close(runner) -> None:
        deferred_runners.append(runner)

    _C.close = defer_close
    sys.argv = [sys.argv[0], "train", ENV_NAME, *sys.argv[1:]]

    try:
        pufferl.main()
    except SystemExit as exc:
        exit_without_native_cleanup(int(exc.code or 0))
    except BaseException:
        traceback.print_exc()
        exit_without_native_cleanup(1)

    exit_without_native_cleanup(0)


if __name__ == "__main__":
    main()
