"""Public result types."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReflectanceJacobian:
    """Reflectance and per-wavelength sensitivities.

    ``d_thickness`` has shape ``(n_wavelengths, n_finite_layers)``.  A caller
    using a wavelength-independent thickness vector must sum its weighted
    rows when applying the chain rule.
    """

    reflectance: np.ndarray
    d_mua: np.ndarray
    d_musp: np.ndarray
    d_thickness: np.ndarray
