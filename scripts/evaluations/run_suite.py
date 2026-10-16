#!/usr/bin/env python3
"""Run isolated MLDT accuracy, performance, and application evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluations.benchmark_core import BACKEND_NAMES, SCHEMA_VERSION, STAGE_NAMES
from scripts.evaluations.benchmark_core.schema import (
    environment,
    read_json,
    stable_hash,
    stage_document,
    validate_stage,
    write_json,
)

PROFILE_DEFAULTS = {
    "smoke": {
        "stages": STAGE_NAMES[:6],
        "warmup": 0,
        "repetitions": 1,
        "settings": {
            "equivalence_workloads": [[4, 1], [8, 3]],
            "robustness_rows": 4,
            "robustness_layers": [1, 3],
            "performance_workloads": [[4, 1], [16, 3]],
            "inference_samples": 20,
            "inference_wavelengths": 5,
            "inference_mini_batch": 10,
            "gradient_rows": 2,
            "gradient_layers": 3,
        },
    },
    "standard": {
        "stages": STAGE_NAMES[:6],
        "warmup": 3,
        "repetitions": 10,
        "settings": {
            "equivalence_workloads": [[1, 1], [32, 10], [96, 20], [201, 101]],
            "robustness_rows": 1000,
            "robustness_layers": list(range(1, 9)),
            "performance_workloads": [[1, 1], [32, 10], [96, 20], [201, 101]],
            "inference_samples": 10_000,
            "inference_wavelengths": 100,
            "inference_mini_batch": 10,
            "gradient_rows": 8,
            "gradient_layers": 3,
        },
    },
    "publication": {
        "stages": STAGE_NAMES[:7],
        "warmup": 10,
        "repetitions": 50,
        "settings": {
            "equivalence_workloads": [[1, 1], [32, 10], [96, 20], [201, 101]],
            "robustness_rows": 2500,
            "robustness_layers": list(range(1, 9)),
            "performance_workloads": [[1, 1], [32, 10], [64, 16], [96, 20], [201, 101]],
            "inference_samples": 10_000,
            "inference_wavelengths": 100,
            "inference_mini_batch": 10,
            "gradient_rows": 32,
            "gradient_layers": 3,
        },
    },
}


def load_dotenv(path: Path) -> None:
    """Load simple key/value settings without overriding the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def comma_list(value: str) -> list[str]:
    """Parse comma-separated CLI identifiers and normalize hyphens."""
    return [item.strip().replace("-", "_") for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse and validate unified-suite command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="standard")
    parser.add_argument("--stages", type=comma_list)
    parser.add_argument("--backends", type=comma_list, default=list(BACKEND_NAMES))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/suite"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dtypes", type=comma_list, default=["float64"])
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--model-dir", default=os.getenv("model_dir"))
    parser.add_argument("--model-version", default=os.getenv("model_version"))
    parser.add_argument("--checkpoint-path", default=os.getenv("checkpoint_path"))
    args = parser.parse_args()
    invalid_stages = sorted(set(args.stages or ()) - set(STAGE_NAMES))
    invalid_backends = sorted(set(args.backends) - set(BACKEND_NAMES))
    invalid_dtypes = sorted(set(args.dtypes) - {"float32", "float64"})
    if invalid_stages or invalid_backends or invalid_dtypes:
        parser.error(
            "; ".join(
                part
                for part in (
                    f"unknown stages: {invalid_stages}" if invalid_stages else "",
                    f"unknown backends: {invalid_backends}" if invalid_backends else "",
                    f"unknown dtypes: {invalid_dtypes}" if invalid_dtypes else "",
                )
                if part
            )
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main() -> None:
    """Run selected stages in subprocesses and consolidate their results."""
    load_dotenv(Path(".env"))
    args = parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    # Resolve profile defaults once so every stage receives the same run settings.
    stages = list(args.stages or defaults["stages"])
    warmup = defaults["warmup"] if args.warmup is None else args.warmup
    repetitions = defaults["repetitions"] if args.repetitions is None else args.repetitions
    if warmup < 0 or repetitions < 1:
        raise SystemExit("warmup must be non-negative and repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir = args.output_dir / "stages"
    configs_dir = args.output_dir / "configs"
    fragments_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)
    stage_payloads = {}
    for stage in stages:
        # Persist each stage's exact configuration beside its result for resuming.
        config = {
            "stage": stage,
            "profile": args.profile,
            "backends": args.backends,
            "seed": args.seed,
            "dtype": args.dtype,
            "dtypes": args.dtypes,
            "warmup": warmup,
            "repetitions": repetitions,
            "settings": defaults["settings"],
            "external_model": {
                "model_dir": args.model_dir,
                "model_version": args.model_version,
                "checkpoint_path": args.checkpoint_path,
            },
        }
        config_path = configs_dir / f"{stage}.json"
        output_path = fragments_dir / f"{stage}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        if args.resume and output_path.is_file():
            try:
                saved = read_json(output_path)
                validate_stage(saved, stage, config)
            except (ValueError, json.JSONDecodeError):
                pass
            else:
                print(f"Reusing {stage}: {output_path}")
                stage_payloads[stage] = saved
                continue
        print(f"Running {stage}...", flush=True)
        # Isolate stages in subprocesses and retain diagnostics if no result is written.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluations.benchmark_core.worker",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            stage_payloads[stage] = read_json(output_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            fallback = stage_document(
                stage,
                config,
                {
                    "status": "failed",
                    "error": f"stage worker failed without a valid result: {exc}",
                    "worker_returncode": completed.returncode,
                    "worker_stdout": completed.stdout,
                    "worker_stderr": completed.stderr,
                },
            )
            write_json(output_path, fallback)
            stage_payloads[stage] = fallback
    # Combine stage fragments into one reproducible suite document.
    run_config = {
        "profile": args.profile,
        "stages": stages,
        "backends": args.backends,
        "seed": args.seed,
        "dtype": args.dtype,
        "dtypes": args.dtypes,
        "warmup": warmup,
        "repetitions": repetitions,
        "settings": defaults["settings"],
        "external_model": {
            "model_dir": args.model_dir,
            "model_version": args.model_version,
            "checkpoint_path": args.checkpoint_path,
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mldt-evaluation-suite",
        "created_at": datetime.now(UTC).isoformat(),
        "config_hash": stable_hash(run_config),
        "config": run_config,
        "environment": environment(),
        "stages": stage_payloads,
    }
    results_path = args.output_dir / "results.json"
    write_json(results_path, result)
    print(results_path)
    if not args.no_analysis:
        # Reports are derived from the consolidated JSON and can be regenerated later.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluations.analyze_suite",
                str(results_path),
                "--output-dir",
                str(args.output_dir),
            ],
            check=True,
        )
    failed = [
        name for name, payload in stage_payloads.items() if payload.get("status") != "complete"
    ]
    if failed:
        raise SystemExit("failed evaluation stage(s): " + ", ".join(failed))


if __name__ == "__main__":
    main()
