"""Train the shared MAPPO collision-avoidance policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def train(args: argparse.Namespace) -> None:
    """Start a new training run or resume an existing one."""
    from orbitzoo.thesis.config import ExperimentConfig
    from orbitzoo.thesis.runtime import create_run_directory, initialize_run_directory
    from orbitzoo.thesis.training.trainer import train as run_training

    try:
        if args.resume:
            run_directory = Path(args.resume).expanduser()
            config = ExperimentConfig.load(run_directory / "config.json")
        else:
            config = ExperimentConfig.load(Path(args.config).expanduser())
            run_directory = (
                initialize_run_directory(Path(args.output).expanduser(), config)
                if args.output
                else create_run_directory(".", config, "mappo")
            )
        print(f"Run directory: {run_directory}", file=sys.stderr, flush=True)
        result = run_training(
            config,
            run_directory,
            resume=bool(args.resume),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        raise SystemExit(f"oz: training failed: {error}") from error

    print(f"Artifacts: {result.run_directory}")
    print(f"Completed updates: {result.completed_updates}")
    print(f"Environment steps: {result.environment_steps}")
    if result.final_metrics:
        print(f"Final mean episode return: {result.final_metrics['mean_episode_return']:.4f}")
        print(f"Final collision rate: {result.final_metrics['collision_rate']:.2%}")


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "train",
        help="train the shared MAPPO collision-avoidance policy",
    )
    command.add_argument(
        "--config",
        default="configs/mappo_smoke.json",
        help="experiment configuration path (default: %(default)s)",
    )
    command.add_argument(
        "--output",
        help="new run directory; existing paths are refused (default: runs/<timestamp>_mappo_seed<N>)",
    )
    command.add_argument(
        "--resume",
        help="existing run directory to continue from its latest checkpoint",
    )
    command.set_defaults(handler=train)
