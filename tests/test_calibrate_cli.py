from pathlib import Path

import pytest

from orbitzoo.cli import build_parser
from orbitzoo.cli import calibrate as calibrate_module
from orbitzoo.thesis.calibration import runner
from orbitzoo.thesis.calibration.runner import (
    CalibrationRunResult,
    CalibrationRunStatus,
)


def test_calibrate_command_defaults_are_registered() -> None:
    args = build_parser().parse_args(["calibrate"])

    assert args.config == "configs/k_dt_calibration.json"
    assert args.output == "runs/k_dt_calibration"


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [
        (
            CalibrationRunStatus.NO_PASSING_CALIBRATION,
            calibrate_module.NO_PASSING_CALIBRATION_EXIT_CODE,
        ),
        (
            CalibrationRunStatus.REJECTED_VALIDATION,
            calibrate_module.VALIDATION_REJECTED_EXIT_CODE,
        ),
    ],
)
def test_calibrate_command_uses_distinct_scientific_rejection_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: CalibrationRunStatus,
    expected_exit_code: int,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_calibration",
        lambda *_args, **_kwargs: CalibrationRunResult(
            output_directory=tmp_path / "run",
            status=status,
            recommendation=None,
            total_runtime_seconds=1.0,
            reference_conjunction_count=0,
            candidate_window_count=0,
        ),
    )
    args = build_parser().parse_args(["calibrate"])

    with pytest.raises(SystemExit) as exit_info:
        args.handler(args)

    assert exit_info.value.code == expected_exit_code
