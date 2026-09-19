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
    print(
        f"{'policy':<16}{'collisions':>11}{'close calls':>13}{'closest m':>11}"
        f"{'shortfall':>11}{'delta-v m/s':>13}{'drift m':>10}"
    )
    for summary in summaries:
        closest = f"{summary.closest_approach_m:.1f}" if summary.closest_approach_m is not None else "-"
        print(
            f"{summary.policy:<16}{summary.collision_rate:>11.2%}{summary.mean_close_approaches:>13.2f}"
            f"{closest:>11}{summary.mean_close_approach_shortfall:>11.3f}"
            f"{summary.mean_delta_v_per_agent_mps:>13.4f}{summary.mean_slot_offset_m:>10.1f}"
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
