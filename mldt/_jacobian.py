"""Backend-independent numerical Jacobian used as a correctness oracle."""

import numpy as np

from ._system import assemble, solve
from ._validation import Inputs, require_positive_absorption
from .types import ReflectanceJacobian


def _step(value: float) -> float:
    """Choose a stable finite-difference step for a positive parameter."""
    # Scale the nominal step with the parameter, retaining an absolute floor
    # for values below one. The cube-root epsilon is a finite-difference
    # heuristic, rather than an error-optimal step for every assembled entry.
    nominal = np.cbrt(np.finfo(np.float64).eps) * max(1.0, abs(value))
    # The stencil reaches value - 2*h; this cap keeps every sample positive.
    return min(nominal, value / 4.0)


def _dense(ab: np.ndarray) -> np.ndarray:
    """Expand batched banded matrices into dense matrices."""
    n = ab.shape[2]
    matrix = np.zeros((ab.shape[0], n, n), dtype=np.float64)
    for row in range(n):
        # Only the main diagonal and two diagonals on each side are stored.
        for col in range(max(0, row - 2), min(n, row + 3)):
            # LAPACK band storage: ab[w, 2 + row - col, col] = A[w, row, col].
            matrix[:, row, col] = ab[:, 2 + row - col, col]
    return matrix


def adjoint(inputs: Inputs) -> ReflectanceJacobian:
    """Compute Jacobians with one implicit adjoint solve per wavelength.

    Stable five-point differences differentiate the assembled matrix and
    right-hand side; they do not repeat the coefficient solve.  The derivative
    then follows from ``lambda.T @ (db - dA @ coefficients)``.
    """
    # Forward solves allow zero absorption, but optical derivatives there
    # are undefined; the centered stencil also requires positive samples.
    require_positive_absorption(inputs.mua)
    base_system = assemble(inputs)
    base, coefficients = solve(base_system)
    matrices = _dense(base_system.ab)
    nw, nl = inputs.mua.shape
    nf = inputs.thicknesses.size
    dm = np.empty_like(inputs.mua)
    ds = np.empty_like(inputs.musp)
    # Thicknesses are shared across wavelengths. A one-layer input has no
    # finite layers, so its thickness Jacobian has shape (nw, 0).
    dd = np.empty((nw, nf), dtype=np.float64)

    adjoints = np.empty_like(coefficients)
    objective = np.zeros(coefficients.shape[1], dtype=np.float64)
    # Rd = boundary_constant * coefficients[0], so this vector is dRd/dc.
    # With A*c = b, solve A.T*lambda = dRd/dc once and reuse lambda for
    # every parameter: dRd/dp = lambda.T * (db/dp - dA/dp * c).
    # Refractive indices stay fixed, so the boundary constant has no derivative.
    objective[0] = base_system.boundary_constant
    for w in range(nw):
        adjoints[w] = np.linalg.solve(matrices[w].T, objective)

    def evaluate_system(
        mua: np.ndarray, musp: np.ndarray, thicknesses: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble one perturbed optical system."""
        # Perturb only assembly: no new forward or adjoint solve is needed.
        changed = Inputs(
            mua, musp, thicknesses, inputs.refractive_index, inputs.outside_refractive_index
        )
        system = assemble(changed)
        return _dense(system.ab), system.rhs

    def contract(
        systems: list[tuple[np.ndarray, np.ndarray]],
        h: float,
        wavelength: int | None = None,
    ) -> np.ndarray | np.float64:
        """Contract finite-difference system derivatives with adjoints."""
        # Samples are ordered (+2h, +h, -h, -2h). This centered five-point
        # stencil has fourth-order truncation error for smooth system entries.
        dmatrix = (-systems[0][0] + 8 * systems[1][0] - 8 * systems[2][0] + systems[3][0]) / (
            12 * h
        )
        drhs = (-systems[0][1] + 8 * systems[1][1] - 8 * systems[2][1] + systems[3][1]) / (12 * h)
        if wavelength is not None:
            # Optical perturbations assemble a batch of size one; its index 0
            # corresponds to the original wavelength's coefficients/adjoint.
            return adjoints[wavelength] @ (drhs[0] - dmatrix[0] @ coefficients[wavelength])
        # Shared thickness perturbations affect every wavelength. First form
        # db - dA*c per wavelength, then take its dot product with the adjoint.
        return np.einsum(
            "wi,wi->w",
            adjoints,
            drhs - np.einsum("wij,wj->wi", dmatrix, coefficients),
        )

    # Wavelength systems are independent: an optical property at (w, k)
    # changes only Rd[w], so only that wavelength needs to be reassembled.
    for w in range(nw):
        for k in range(nl):
            h = _step(inputs.mua[w, k])
            systems = []
            for multiplier in (2.0, 1.0, -1.0, -2.0):
                # Copy the perturbed array to preserve the validated inputs.
                changed_mua = inputs.mua[w : w + 1].copy()
                changed_mua[0, k] += multiplier * h
                systems.append(
                    evaluate_system(
                        changed_mua,
                        inputs.musp[w : w + 1],
                        inputs.thicknesses,
                    )
                )
            dm[w, k] = contract(systems, h, w)

            h = _step(inputs.musp[w, k])
            systems = []
            for multiplier in (2.0, 1.0, -1.0, -2.0):
                changed_musp = inputs.musp[w : w + 1].copy()
                changed_musp[0, k] += multiplier * h
                systems.append(
                    evaluate_system(
                        inputs.mua[w : w + 1],
                        changed_musp,
                        inputs.thicknesses,
                    )
                )
            ds[w, k] = contract(systems, h, w)

    # Each finite-layer thickness is common to all wavelength systems, so
    # assemble the full batch and fill one column of dd per perturbation set.
    for k in range(nf):
        h = _step(inputs.thicknesses[k])
        systems = []
        for multiplier in (2.0, 1.0, -1.0, -2.0):
            changed = inputs.thicknesses.copy()
            changed[k] += multiplier * h
            systems.append(evaluate_system(inputs.mua, inputs.musp, changed))
        dd[:, k] = contract(systems, h)
    return ReflectanceJacobian(base, dm, ds, dd)


# Kept private to the implementation history: callers use the adjoint above.
finite_difference = adjoint
