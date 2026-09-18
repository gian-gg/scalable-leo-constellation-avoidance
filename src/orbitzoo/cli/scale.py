"""Measure how the frozen decentralized actor behaves as the catalog and agent count grow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def scale(args: argparse.Namespace) -> None:
    """Run the scalability sweeps and print one line per scenario and policy."""
    from orbitzoo.thesis.scalability.runner import SWEEPS, run_scalability

    output = Path(args.output).expanduser()
    try:
        rows = run_scalability(
            Path(args.config).expanduser(),
            args.policy or ["rule"],
            output,
            sweeps=SWEEPS if args.sweep == "both" else (args.sweep,),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        raise SystemExit(f"oz: scalability evaluation failed: {error}") from error

    print(f"Artifacts: {output}")
    print(f"{'sweep':<9}{'objects':>9}{'agents':>8}  {'policy':<12}{'conj.':>7}{'second.':>9}{'s/decision':>12}")
    for row in rows:
        seconds = sum(float(value) for key, value in row.items() if key.endswith("_seconds_per_decision"))
        print(
            f"{row['sweep']:<9}{row['objects']:>9}{row['agents']:>8}  {row['policy']:<12}"
            f"{row['conjunctions']:>7}{str(row['secondary_conjunctions']):>9}{seconds:>12.3f}"
        )


def add_scale_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "scale",
        help="evaluate the frozen actor at catalog scale",
    )
    command.add_argument(
        "--config",
        default="configs/scalability_smoke.json",
        help="scalability configuration path (default: %(default)s)",
    )
    command.add_argument(
        "--policy",
        action="append",
        help="rule or [NAME=]PATH to a MAPPO checkpoint; repeatable; no-op always runs (default: rule)",
    )
    command.add_argument(
        "--sweep",
        choices=("catalog", "agents", "both"),
        default="both",
        help="which sweep to run (default: %(default)s)",
    )
    command.add_argument(
        "--output",
        required=True,
        help="new artifact directory; existing paths are refused",
    )
    command.set_defaults(handler=scale)
