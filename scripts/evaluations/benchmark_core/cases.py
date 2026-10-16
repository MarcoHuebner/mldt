"""Deterministic workloads for correctness and performance evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

REGIMES = ("tissue", "broad", "low_absorption", "resonant", "zero_absorption", "thick")


def log_uniform(
    rng: np.random.Generator, low: float, high: float, shape: tuple[int, ...]
) -> np.ndarray:
    """Draw a deterministic log-uniform array from the supplied generator."""
    return np.exp(rng.uniform(math.log(low), math.log(high), shape))


def ordinary_case(
    rng: np.random.Generator, rows: int, layers: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create a well-conditioned workload in the original notebook's domain."""
    return (
        rng.uniform(0.001, 0.05, (rows, layers)),
        rng.uniform(0.5, 3.0, (rows, layers)),
        rng.uniform(0.01, 1.0, layers - 1),
        1.4,
        1.0,
    )


def regime_case(
    rng: np.random.Generator, rows: int, layers: int, regime: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create one deterministic numerical-robustness workload.

    ``mua`` and ``musp`` are sampled per row and layer. ``thicknesses`` is one
    sampled layer stack shared by all rows in this batch. Positive ranges use
    log-uniform sampling so values are spread across orders of magnitude.

    Regimes:
      - ``tissue``: tissue-like absorption, scattering, and layer thicknesses.
      - ``broad``: very broad property and thickness ranges, including extreme
        scales that stress numerical range and conditioning.
      - ``low_absorption``: near-zero positive absorption with ordinary
        scattering, testing the low-absorption limit.
      - ``resonant``: scattering is approximately twice absorption, with tiny
        offsets and exact matches every tenth row to probe a resonant formula.
      - ``zero_absorption``: exactly zero absorption, testing the analytic
        zero-absorption limit.
      - ``thick``: very thick layers, testing optically thick behavior and
        underflow/overflow sensitivity.
    """
    # Each batch shares a layer stack while optical properties vary by row.
    shape = (rows, layers)
    if regime == "tissue":
        mua = log_uniform(rng, 1e-4, 0.2, shape)
        musp = log_uniform(rng, 0.2, 5.0, shape)
        thicknesses = log_uniform(rng, 0.01, 10.0, layers - 1)
    elif regime == "broad":
        mua = log_uniform(rng, 1e-8, 10.0, shape)
        musp = log_uniform(rng, 1e-4, 30.0, shape)
        thicknesses = log_uniform(rng, 1e-5, 1e3, layers - 1)
    elif regime == "low_absorption":
        mua = log_uniform(rng, 1e-14, 1e-5, shape)
        musp = log_uniform(rng, 0.05, 10.0, shape)
        thicknesses = log_uniform(rng, 1e-4, 100.0, layers - 1)
    elif regime == "resonant":
        mua = log_uniform(rng, 1e-4, 5.0, shape)
        offsets = rng.uniform(-1e-7, 1e-7, shape)
        offsets[::10] = 0.0
        musp = 2.0 * mua * (1.0 + offsets)
        thicknesses = log_uniform(rng, 1e-4, 100.0, layers - 1)
    elif regime == "zero_absorption":
        mua = np.zeros(shape)
        musp = log_uniform(rng, 0.05, 10.0, shape)
        thicknesses = log_uniform(rng, 1e-4, 100.0, layers - 1)
    elif regime == "thick":
        mua = log_uniform(rng, 1e-5, 0.5, shape)
        musp = log_uniform(rng, 0.1, 30.0, shape)
        thicknesses = log_uniform(rng, 10.0, 1e5, layers - 1)
    else:
        raise ValueError(f"unknown regime: {regime}")
    return mua, musp, thicknesses, 1.4, 1.0


def high_layer_case(
    rng: np.random.Generator, profile: str
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """Create the paper or notebook wavelength grid with 101 layers."""
    wavelengths = np.arange(400, 801, 2) if profile == "paper" else np.arange(380, 1001)
    layers = 101
    depth = np.linspace(0, 1, layers)
    absorption = 0.004 + 0.02 * np.exp(-(((depth - 0.35) / 0.18) ** 2))
    absorption += rng.uniform(0, 0.001, layers)
    scattering = 1.0 + 2.5 * np.exp(-(((depth - 0.65) / 0.3) ** 2))
    mua = (0.7 + 0.3 * (500 / wavelengths)[:, None]) * absorption
    musp = scattering[None, :] * (wavelengths[:, None] / 500) ** -2
    thicknesses = np.full(layers - 1, 3.0 / (layers - 1))
    # Return wavelengths separately for labeling; solver inputs omit that metadata.
    return wavelengths, (mua, musp, thicknesses, 1.4, 1.0)


def spectral_batches(samples: int, wavelengths: int, spectra_per_batch: int) -> Iterator[slice]:
    """Yield flat row slices containing complete spectra only."""
    if min(samples, wavelengths, spectra_per_batch) < 1:
        raise ValueError("batch dimensions must be positive")
    # Chunk whole spectra while keeping slices in the flattened input layout.
    width = wavelengths * spectra_per_batch
    for start in range(0, samples * wavelengths, width):
        yield slice(start, min(start + width, samples * wavelengths))
