"""Compare collision-avoidance policies on identical held-out episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate the requested policies and print a summary table."""
    from orbitzoo.thesis.config import ExperimentConfig
    from orbitzoo.thesis.evaluation.evaluator import evaluate as run_evaluation

    try:
        config = ExperimentConfig.load(Path(args.config).expanduser())
        summaries = run_evaluation(
            config,
            args.policy or ["noop", "rule"],
            args.episodes,
            Path(args.output).expanduser(),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        raise SystemExit(f"oz: evaluation failed: {error}") from error

    print(f"Artifacts: {Path(args.output).expanduser()}")
    print(f"{'policy':<16}{'collisions':>12}{'unsafe steps':>14}{'delta-v m/s':>14}{'min sep m':>12}")
    for summary in summaries:
        print(
            f"{summary.policy:<16}{summary.collision_rate:>12.2%}"
            f"{summary.mean_unsafe_agent_steps:>14.2f}"
            f"{summary.mean_delta_v_per_agent_mps:>14.4f}"
            f"{summary.minimum_separation_meters:>12.1f}"
        )


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "evaluate",
        help="compare policies on identical held-out episodes",
    )
    command.add_argument(
        "--config",
        default="configs/mappo_smoke.json",
        help="experiment configuration path (default: %(default)s)",
    )
    command.add_argument(
        "--policy",
        action="append",
        help="noop, rule, or [NAME=]PATH to a MAPPO checkpoint; repeatable (default: noop and rule)",
    )
    command.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="held-out episodes per policy (default: %(default)s)",
    )
    command.add_argument(
        "--output",
        required=True,
        help="new artifact directory; existing paths are refused",
    )
    command.set_defaults(handler=evaluate)
