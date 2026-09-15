"""End-to-end offline k/delta-t calibration command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

NO_PASSING_CALIBRATION_EXIT_CODE = 2
VALIDATION_REJECTED_EXIT_CODE = 3


def calibrate(args: argparse.Namespace) -> None:
    """Run the complete propagation-only calibration workflow."""
    from orbitzoo.thesis.calibration.runner import (
        CalibrationRunStatus,
        run_calibration,
    )

    config_path = Path(args.config).expanduser()
    output_directory = Path(args.output).expanduser()
    try:
        result = run_calibration(
            config_path,
            output_directory,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except Exception as error:
        raise SystemExit(f"oz: calibration failed: {error}") from error

    print(f"Calibration status: {result.status.value}")
    print(f"Artifacts: {result.output_directory}")
    print(f"Reference conjunctions: {result.reference_conjunction_count}")
    print(f"Runtime seconds: {result.total_runtime_seconds:.2f}")
    if result.recommendation is not None:
        print(f"Selected k: {result.recommendation.neighborhood_size}")
        print(
            "Selected delta-t seconds: "
            f"{result.recommendation.decision_interval_seconds}"
        )

    if result.status is CalibrationRunStatus.NO_PASSING_CALIBRATION:
        raise SystemExit(NO_PASSING_CALIBRATION_EXIT_CODE)
    if result.status is CalibrationRunStatus.REJECTED_VALIDATION:
        raise SystemExit(VALIDATION_REJECTED_EXIT_CODE)


def add_calibrate_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "calibrate",
        help="determine k and delta-t from a fixed TLE catalog",
    )
    command.add_argument(
        "--config",
        default="configs/k_dt_calibration.json",
        help="calibration configuration path (default: %(default)s)",
    )
    command.add_argument(
        "--output",
        default="runs/k_dt_calibration",
        help=(
            "new artifact directory; existing paths are refused "
            "(default: %(default)s)"
        ),
    )
    command.set_defaults(handler=calibrate)
