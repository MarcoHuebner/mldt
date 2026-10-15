"""Calculate the Fresnel partial-current boundary constant."""

import math
from functools import lru_cache

from scipy.integrate import quad


def _positive_finite(value: float, name: str) -> float:
    """Convert a value to a positive finite float."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


@lru_cache(maxsize=128)
def partial_current_constant(n_tissue: float, n_outside: float = 1.0) -> float:
    """Return the exact Fresnel partial-current constant ``A``.

    SciPy's adaptive QUADPACK integration is split at the critical angle when
    total internal reflection is present.
    """

    n_tissue = _positive_finite(n_tissue, "n_tissue")
    n_outside = _positive_finite(n_outside, "n_outside")
    ratio = n_outside / n_tissue

    def fresnel(theta: float) -> float:
        """Return unpolarized Fresnel reflectance at an incident angle."""
        sine = math.sin(theta)
        if ratio < 1.0 and sine >= ratio:
            return 1.0
        transmitted = math.asin(min(1.0, (n_tissue / n_outside) * sine))
        ci, ct = math.cos(theta), math.cos(transmitted)
        rs = ((n_tissue * ci - n_outside * ct) / (n_tissue * ci + n_outside * ct)) ** 2
        rp = ((n_outside * ci - n_tissue * ct) / (n_outside * ci + n_tissue * ct)) ** 2
        return 0.5 * (rs + rp)

    def fluence_moment(theta: float) -> float:
        """Return the angularly weighted Fresnel fluence integrand."""
        return 2.0 * math.sin(theta) * math.cos(theta) * fresnel(theta)

    def current_moment(theta: float) -> float:
        """Return the angularly weighted Fresnel current integrand."""
        return 3.0 * math.sin(theta) * math.cos(theta) ** 2 * fresnel(theta)

    points = [math.asin(ratio)] if ratio < 1.0 else None
    r_phi = quad(fluence_moment, 0.0, math.pi / 2.0, points=points, epsabs=1e-12, epsrel=1e-12)[0]
    r_j = quad(current_moment, 0.0, math.pi / 2.0, points=points, epsabs=1e-12, epsrel=1e-12)[0]
    return (1.0 - r_phi) / (2.0 * (1.0 + r_j))
