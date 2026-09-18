"""Top-level parser and entry point for the ``oz`` command."""

import argparse

from orbitzoo.cli.calibrate import add_calibrate_parser
from orbitzoo.cli.catalog import add_catalog_parser
from orbitzoo.cli.demo import add_demo_parser
from orbitzoo.cli.evaluate import add_evaluate_parser
from orbitzoo.cli.missions import add_mission_parsers
from orbitzoo.cli.run_file import add_run_parser
from orbitzoo.cli.train import add_train_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oz",
        description="Run OrbitZoo tools, demos, and mission scripts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_demo_parser(subparsers)
    add_run_parser(subparsers)
    add_mission_parsers(subparsers)
    add_catalog_parser(subparsers)
    add_calibrate_parser(subparsers)
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
