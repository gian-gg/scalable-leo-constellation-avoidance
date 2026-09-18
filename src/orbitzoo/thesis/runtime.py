"""Run-directory and device utilities shared by thesis experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING

from orbitzoo.thesis.config import ExperimentConfig

if TYPE_CHECKING:
    import torch


def _torch():
    """Import PyTorch only when runtime/device functionality is requested."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to select a training device or create a run directory. "
            "Install the project's training dependencies first."
        ) from error
    return torch


def select_device() -> "torch.device":
    """Prefer CUDA, then Apple Metal (MPS), otherwise use CPU."""
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info(device: "torch.device | None" = None) -> dict[str, str | int | bool | None]:
    """Return execution metadata to save beside every experiment."""
    torch = _torch()
    selected = device or select_device()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
        "device": str(selected),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }


def create_run_directory(
    project_root: str | Path,
    config: ExperimentConfig,
    label: str,
    now: datetime | None = None,
) -> Path:
    """Create a self-describing run directory and save reproducibility metadata."""
    config.validate()
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H%M%SZ")
    safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
    run_directory = Path(project_root) / "runs" / f"{timestamp}_{safe_label}_seed{config.seed}"
    return initialize_run_directory(run_directory, config)


def initialize_run_directory(run_directory: str | Path, config: ExperimentConfig) -> Path:
    """Create a new run directory holding the config and execution metadata."""
    config.validate()
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=False)
    config.save(run_directory / "config.json")
    (run_directory / "tensorboard").mkdir()
    (run_directory / "environment_info.json").write_text(
        json.dumps(environment_info(), indent=2, sort_keys=True) + "\n"
    )
    return run_directory
