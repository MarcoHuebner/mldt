"""Versioned result I/O and environment metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import SCHEMA_VERSION


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable value."""
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict, indented JSON and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read an object using the suite's currently supported schema version."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version in {path}: {value.get('schema_version')!r}")
    return value


def environment() -> dict[str, Any]:
    """Capture dependency, hardware, threading, and Git metadata."""
    # Optional hardware and Git details enrich provenance without blocking a run.
    torch_version = None
    cuda_device = None
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch_version,
        "cuda_device": cuda_device,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def stage_document(stage: str, config: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Build a versioned stage result with a configuration hash."""
    # Hashing the config lets resume logic reject stale stage output.
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mldt-evaluation-stage",
        "stage": stage,
        "config_hash": stable_hash(config),
        "config": config,
        **body,
    }


def validate_stage(payload: dict[str, Any], stage: str, config: dict[str, Any]) -> None:
    """Validate that a saved stage can be resumed for this configuration."""
    if payload.get("kind") != "mldt-evaluation-stage" or payload.get("stage") != stage:
        raise ValueError(f"result is not a {stage!r} stage document")
    if payload.get("config_hash") != stable_hash(config):
        raise ValueError(f"saved {stage!r} result does not match the requested configuration")
