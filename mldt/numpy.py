"""Canonical NumPy/SciPy MLDT backend."""

import numpy as np
import numpy.typing as npt

from ._system import reflectance_validated
from ._validation import validate_numpy


def reflectance(
    mua: npt.ArrayLike,
    musp: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> np.ndarray:
    """Compute diffuse reflectance for one or more wavelengths."""
    inputs = validate_numpy(mua, musp, thicknesses, refractive_index, outside_refractive_index)
    return reflectance_validated(inputs)
