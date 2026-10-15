"""Optional compiled backend and explicit Jacobian API."""

import numpy as np
import numpy.typing as npt

from ._jacobian import adjoint
from ._validation import validate_numpy
from .boundary import partial_current_constant
from .types import ReflectanceJacobian

try:
    from ._native import core as _core
except ImportError as exc:  # pragma: no cover - environment dependent
    _core = None
    _native_error = exc
else:
    _native_error = None


def is_available() -> bool:
    """Report whether the optional native backend can be imported."""
    return _core is not None


def _require_native() -> None:
    """Load the native backend or raise an actionable installation error."""
    if _core is None:
        raise ImportError(
            "the optional MLDT native extension is not installed; reinstall "
            "with native build support or use mldt.numpy.reflectance"
        ) from _native_error


def reflectance(
    mua: npt.ArrayLike,
    musp: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> np.ndarray:
    """Compute reflectance with the optional native backend."""
    # Fail early when the optional extension is unavailable, then normalize
    # inputs through the same validation path as the NumPy backend.
    _require_native()
    inputs = validate_numpy(mua, musp, thicknesses, refractive_index, outside_refractive_index)
    # The native kernel receives the scalar boundary constant, not indices.
    boundary = partial_current_constant(inputs.refractive_index, inputs.outside_refractive_index)
    return _core.reflectance(
        inputs.mua,
        inputs.musp,
        inputs.thicknesses,
        boundary,
    )


def reflectance_jacobian(
    mua: npt.ArrayLike,
    musp: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> ReflectanceJacobian:
    """Compute reflectance and optical-property Jacobians."""
    # Reuse shared validation and the backend-independent adjoint oracle.
    inputs = validate_numpy(mua, musp, thicknesses, refractive_index, outside_refractive_index)
    return adjoint(inputs)
