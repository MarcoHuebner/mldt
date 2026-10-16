import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluations.analyze_suite import analyze
from scripts.evaluations.benchmark_core.backends import discover
from scripts.evaluations.benchmark_core.cases import spectral_batches
from scripts.evaluations.benchmark_core.metrics import error_summary, measure
from scripts.evaluations.benchmark_core.schema import (
    read_json,
    stage_document,
    validate_stage,
    write_json,
)


def test_spectral_batches_keep_complete_spectra_together() -> None:
    assert list(spectral_batches(23, 7, 10)) == [
        slice(0, 70),
        slice(70, 140),
        slice(140, 161),
    ]


@pytest.mark.parametrize(
    ("samples", "wavelengths", "batch_size"),
    [(0, 7, 2), (2, 0, 2), (2, 7, 0)],
)
def test_spectral_batches_reject_nonpositive_dimensions(
    samples: int, wavelengths: int, batch_size: int
) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        list(spectral_batches(samples, wavelengths, batch_size))


def test_error_summary_counts_failures_and_nonfinite_values() -> None:
    summary = error_summary([1.0, 2.1, np.nan], [1.0, 2.0, 3.0], atol=0.01, rtol=0)
    assert summary["count"] == 3
    assert summary["tolerance_failures"] == 2
    assert summary["nonfinite_mismatches"] == 1
    assert summary["max_abs_error"] == pytest.approx(0.1)


def test_error_summary_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        error_summary([1.0, 2.0], [1.0], atol=0.01, rtol=0)


def test_measure_retains_raw_samples() -> None:
    timing = measure(lambda: None, warmup=1, repetitions=3)
    assert len(timing["samples"]) == 3
    assert timing["q1"] <= timing["median"] <= timing["q3"]


def test_backend_discovery_reports_unavailable_cuda() -> None:
    available, unavailable = discover(["numpy", "torch_gpu"])
    assert "numpy" in available
    assert ("torch_gpu" in available) != ("torch_gpu" in unavailable)


def test_stage_schema_rejects_mismatched_resume_configuration(tmp_path: Path) -> None:
    config = {"stage": "equivalence", "seed": 1}
    path = tmp_path / "stage.json"
    write_json(path, stage_document("equivalence", config, {"status": "complete"}))
    payload = read_json(path)
    validate_stage(payload, "equivalence", config)
    with pytest.raises(ValueError, match="does not match"):
        validate_stage(payload, "equivalence", {**config, "seed": 2})


def test_analyzer_rejects_incomplete_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a unified"):
        analyze({"kind": "wrong"}, tmp_path)


def test_schema_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("[]")
    with pytest.raises(TypeError, match="JSON object"):
        read_json(path)


def test_smoke_profile_runs_all_non_external_stages_and_analysis(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    command = [
        sys.executable,
        "scripts/evaluations/run_suite.py",
        "--profile",
        "smoke",
        "--backends",
        "original,numpy,cython,torch-cpu",
        "--output-dir",
        str(output),
    ]
    subprocess.run(command, check=True)
    payload = json.loads((output / "results.json").read_text())
    assert set(payload["stages"]) == {
        "equivalence",
        "robustness",
        "performance",
        "inference",
        "high-layer",
        "gradients",
    }
    assert all(stage["status"] == "complete" for stage in payload["stages"].values())
    assert (output / "report.md").is_file()
    assert (output / "inference.png").is_file()
    resumed = subprocess.run(
        [*command, "--resume", "--no-analysis"], check=True, text=True, capture_output=True
    )
    assert resumed.stdout.count("Reusing ") == 6


def test_worker_serializes_stage_failure(tmp_path: Path) -> None:
    config = {
        "stage": "equivalence",
        "profile": "smoke",
        "backends": ["numpy"],
        "seed": 1,
        "dtype": "float64",
        "dtypes": ["float64"],
        "warmup": 0,
        "repetitions": 1,
        "settings": {"equivalence_workloads": [[1, 1]]},
    }
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
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
    )
    assert result.returncode != 0
    assert read_json(output_path)["status"] == "failed"
