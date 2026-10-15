"""Shared array and physical-domain validation."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Inputs:
    """Validated float64 optical inputs shared by NumPy-based solvers."""

    mua: np.ndarray
    musp: np.ndarray
    thicknesses: np.ndarray
    refractive_index: float
    outside_refractive_index: float


def validate_numpy(
    mua: npt.ArrayLike,
    musp: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    refractive_index: float,
    outside_refractive_index: float = 1.0,
) -> Inputs:
    """Validate and normalize NumPy optical-property inputs."""
    mua = np.asarray(mua, dtype=np.float64)
    musp = np.asarray(musp, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    if mua.ndim != 2 or musp.ndim != 2:
        raise ValueError("mua and musp must be 2-D (n_wavelengths, n_layers)")
    if mua.shape != musp.shape:
        raise ValueError(f"mua shape {mua.shape} does not match musp shape {musp.shape}")
    if mua.shape[0] < 1 or mua.shape[1] < 1:
        raise ValueError("mua and musp must contain at least one wavelength and layer")
    if thicknesses.ndim != 1:
        raise ValueError("thicknesses must be a 1-D array")
    expected = mua.shape[1] - 1
    if thicknesses.size != expected:
        raise ValueError(f"thicknesses must contain {expected} values; got {thicknesses.size}")
    for array, name in ((mua, "mua"), (musp, "musp"), (thicknesses, "thicknesses")):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(mua < 0.0):
        raise ValueError("mua must be non-negative")
    if np.any(musp <= 0.0):
        raise ValueError("musp must be strictly positive")
    if np.any(thicknesses <= 0.0):
        raise ValueError("thicknesses must be strictly positive")
    nr = float(refractive_index)
    no = float(outside_refractive_index)
    if not math.isfinite(nr) or nr <= 0.0:
        raise ValueError("refractive_index must be a positive finite scalar")
    if not math.isfinite(no) or no <= 0.0:
        raise ValueError("outside_refractive_index must be a positive finite scalar")
    return Inputs(
        np.ascontiguousarray(mua),
        np.ascontiguousarray(musp),
        np.ascontiguousarray(thicknesses),
        nr,
        no,
    )


def require_positive_absorption(mua: np.ndarray) -> None:
    """Reject zero absorption where optical-property gradients are undefined."""
    if np.any(mua == 0.0):
        raise ValueError(
            "optical-property gradients are undefined at zero absorption; "
            "forward reflectance remains supported"
        )
