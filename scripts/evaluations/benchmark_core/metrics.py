"""Accuracy and timing metrics shared by evaluation stages."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def error_summary(actual: Any, reference: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    """Summarize finite/non-finite errors and tolerance failures."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {reference.shape}")
    # Separate finite errors from non-finite mismatches so both appear in the summary.
    finite = np.isfinite(actual) & np.isfinite(reference)
    mismatch = np.isfinite(actual) != np.isfinite(reference)
    errors = np.full(reference.shape, np.inf)
    errors[finite] = np.abs(actual[finite] - reference[finite])
    relative = np.full(reference.shape, np.inf)
    relative[finite] = errors[finite] / np.maximum(np.abs(reference[finite]), np.finfo(float).tiny)
    failures = errors > atol + rtol * np.abs(reference)
    finite_errors = errors[np.isfinite(errors)]
    finite_relative = relative[np.isfinite(relative)]
    worst = int(np.argmax(errors)) if errors.size else None
    return {
        "count": int(errors.size),
        "tolerance_failures": int(np.count_nonzero(failures)),
        "nonfinite_mismatches": int(np.count_nonzero(mismatch)),
        "mean_abs_error": float(np.mean(finite_errors)) if finite_errors.size else None,
        "p99_abs_error": float(np.quantile(finite_errors, 0.99)) if finite_errors.size else None,
        "p999_abs_error": float(np.quantile(finite_errors, 0.999)) if finite_errors.size else None,
        "max_abs_error": float(np.max(finite_errors)) if finite_errors.size else None,
        "max_rel_error": float(np.max(finite_relative)) if finite_relative.size else None,
        "worst_index": worst,
    }


def measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure synchronized wall time and retain every raw sample."""
    synchronize = synchronize or (lambda: None)
    # Warm up first, then synchronize around samples (needed for asynchronous devices).
    for _ in range(warmup):
        function()
    synchronize()
    samples = []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(samples)
    return {
        "unit": "milliseconds",
        "samples": samples,
        "median": statistics.median(samples),
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }
