"""Size the discrete maneuver delta-v from the calibration's reference conjunctions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def size_maneuvers(args: argparse.Namespace) -> None:
    """Run the sizing study and print the recommendation."""
    from orbitzoo.thesis.maneuvers.sizing_study import run_sizing

    output = Path(args.output).expanduser()
    try:
        run_sizing(
            Path(args.config).expanduser(),
            output,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        raise SystemExit(f"oz: maneuver sizing failed: {error}") from error

    recommendation = json.loads((output / "recommendation.json").read_text())
    print(f"Artifacts: {output}")
    print(f"Rule: {recommendation['rule']}")
    print(f"Selected delta-v per burn (m/s): {recommendation['selected_delta_v_per_burn_mps']}")
    for mass, thrust in recommendation["minimum_thrust_newtons_by_mass_kg"].items():
        print(f"Minimum thrust for a {mass} kg satellite (N): {thrust}")
    print(f"Orekit spot-checks passed: {recommendation['spot_checks_passed']}/{recommendation['spot_checks']}")


def add_size_maneuvers_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "size-maneuvers",
        help="size the maneuver delta-v from reference conjunctions",
    )
    command.add_argument(
        "--config",
        default="configs/maneuver_sizing.json",
        help="sizing configuration path (default: %(default)s)",
    )
    command.add_argument(
        "--output",
        required=True,
        help="new artifact directory; existing paths are refused",
    )
    command.set_defaults(handler=size_maneuvers)
