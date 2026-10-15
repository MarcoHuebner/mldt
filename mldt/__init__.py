"""Stable multi-layer diffusion-theory reflectance solvers for planar media.

Backend implementations live in :mod:`mldt.numpy`, :mod:`mldt.cython`, and
:mod:`mldt.torch`. The public boundary helper supports different tissue and
outside refractive indices through the partial-current Fresnel condition.
"""

from .boundary import partial_current_constant
from .types import ReflectanceJacobian

__all__ = ["ReflectanceJacobian", "partial_current_constant"]
__version__ = "0.1.0"
