"""Stable MLDT system assembly and NumPy reference solve.

Depth z increases into the medium and is measured from each layer's top.
For unit incident illumination, write fluence as phi(z) = H(z) + P(z), where
H solves the homogeneous diffusion equation and P is a source particular
solution. With mu = mua + musp, D = 1/(3*mu), lam = sqrt(mua/D), and
F = cumulative source attenuation * musp/D, choose

    P(z) = F * (exp(-mu*z) - exp(-lam*z)) / (lam**2 - mu**2).

This shifts a homogeneous term from the original particular solution into H.
At resonance (lam == mu), evaluate P by its continuous limit. In each finite
layer the unknowns are u = H(0), v = H(d); the final semi-infinite layer has
only H(z) = u*exp(-lam*z). This basis avoids growing exponential amplitudes.

Current is J = -D*phi'. The surface condition is D*phi'(0) = A*phi(0), and
both phi and J are continuous at internal interfaces. Since P(0) = 0,
surface fluence is the first unknown u0, giving reflectance Rd = A*u0.
Here A is the Fresnel partial-current constant, not an extrapolation factor.

See demoMLDT.ipynb and mldt/original.py for the original amplitude formulation
and README.md's citation for the MLDT publication. The endpoint basis above
documents this implementation's reformulation; equivalence assumes the
original's clipping and denominator safeguards are inactive.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, solve_banded

from ._validation import Inputs
from .boundary import partial_current_constant

# Performance heuristics, not accuracy tolerances; retune for the target hardware.
# Largest system eligible for batched dense solving.
_DENSE_SOLVE_MAX_COEFFICIENTS = 15
# Minimum wavelength count for batched dense solving.
_DENSE_SOLVE_MIN_BATCH = 16
# Dense matrix storage budget per chunk (64 MiB).
_DENSE_SOLVE_MEMORY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BandedSystem:
    """Batched five-diagonal systems and their boundary constant.

    For W wavelengths, L layers, and N = 2*L - 1 unknowns, ``ab`` has shape
    (W, 5, N), with matrix entry M[i, j] stored at ab[w, 2+i-j, j].
    ``rhs`` has shape (W, N). Unknowns are ordered [u0, v0, ..., u_last].
    ``boundary_constant`` is the scalar Fresnel constant A shared by wavelengths.
    """

    ab: np.ndarray
    rhs: np.ndarray
    boundary_constant: float


def _divided_exponential(
    lam: np.ndarray, mu: np.ndarray, distance: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Return P/F and P'/F for broadcast-compatible decay rates and distances.

    ``lam`` and ``mu`` are nonnegative diffusion and positive transport rates;
    ``distance`` is nonnegative local depth. Both outputs have their broadcast
    shape, normally (n_wavelengths, n_finite_layers).
    """
    # P = F * (exp(-mu*z) - exp(-lam*z)) / (lam**2 - mu**2).
    # Subtracting a homogeneous term from the original source makes P(0) = 0;
    # the shifted homogeneous amplitude preserves the same total fluence.
    # Broadcast all layer and wavelength values to one common evaluation shape.
    gap = lam - mu
    scale = lam + mu
    z = np.broadcast_to(distance, np.broadcast_shapes(gap.shape, np.shape(distance)))
    gap = np.broadcast_to(gap, z.shape)
    mu = np.broadcast_to(mu, z.shape)
    scale = np.broadcast_to(scale, z.shape)
    magnitude = np.abs(gap)
    az = magnitude * z
    # The cubic relative remainder is O((g*z)**3/24), below float64 precision
    # at this cutoff; the series also evaluates the exact g = 0 limit.
    small = az <= 1e-7
    # Evaluate the divided difference directly except near coincident rates,
    # where its Taylor expansion avoids cancellation and resolves resonance.
    # At lam = mu > 0, P/F tends to z * exp(-lam*z) / (2*lam).
    safe_gap = np.where(magnitude == 0.0, 1.0, magnitude)
    h = np.empty_like(az)
    regular = ~small
    h[regular] = -np.expm1(-az[regular]) / safe_gap[regular]
    # Taylor expansion of (1 - exp(-g*z))/g for small g*z, where g = |lam - mu|.
    # At resonance (g = 0), this gives the finite limit h = z.
    zs, gaps = z[small], magnitude[small]
    h[small] = zs - gaps * zs * zs / 2.0 + gaps * gaps * zs * zs * zs / 6.0
    # Restore the shared exponential factor and its spatial derivative.
    base = np.minimum(lam, mu)
    exp_base = np.exp(-base * z)
    exp_gap = np.exp(-az)
    value = exp_base * h / scale
    derivative = exp_base * (exp_gap - base * h) / scale
    return value, derivative


def _endpoint_rates(lam: np.ndarray, thicknesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint derivative rates a and b for finite layers.

    ``lam`` has shape (n_wavelengths, n_finite_layers), and positive
    ``thicknesses`` has shape (n_finite_layers,). Both outputs match ``lam``.
    At lam = 0, both rates tend to 1/thickness.
    """
    # For homogeneous endpoint values u = H(0), v = H(d), the derivatives are
    # H'(0) = -a*u + b*v and H'(d) = -b*u + a*v,
    # with a = lam*coth(lam*d), b = lam/sinh(lam*d).
    x = lam * thicknesses[None, :]
    # The retained series through x**4 has relative error O(x**6), below
    # float64 precision at this cutoff, and includes the zero-absorption limit.
    small = np.abs(x) < 1e-4
    # Use series expansions for optically thin layers; use expm1 elsewhere.
    x2 = x * x
    a_small = (1.0 + x2 / 3.0 - x2 * x2 / 45.0) / thicknesses
    b_small = (1.0 - x2 / 6.0 + 7.0 * x2 * x2 / 360.0) / thicknesses
    # This exponential form is stable for both ordinary and optically thick
    # layers and avoids evaluating separate sinh/tanh and large-x branches.
    denom = -np.expm1(-2.0 * x)
    safe_denom = np.where(denom == 0.0, 1.0, denom)
    decay = np.exp(-x)
    a_regular = lam * (1.0 + decay * decay) / safe_denom
    b_regular = lam * 2.0 * decay / safe_denom
    a = np.where(small, a_small, a_regular)
    b = np.where(small, b_small, b_regular)
    return a, b


def assemble(inputs: Inputs) -> BandedSystem:
    """Assemble surface and interface equations from validated optical inputs.

    Inputs contain (n_wavelengths, n_layers) optical arrays and n_layers - 1
    positive finite-layer thicknesses. The last layer extends to infinite depth.
    The returned system uses the endpoint ordering documented by BandedSystem.
    """
    # Derive attenuation, diffusion, and the forcing at each layer boundary.
    mua, musp, d = inputs.mua, inputs.musp, inputs.thicknesses
    nw, nl = mua.shape
    nf = nl - 1
    ncoef = 2 * nf + 1
    # Original notation: mutr = mutr_all, diffusion = zeta_all, lam = 1/delta_all
    # when the original's clipping is inactive.
    mutr = mua + musp
    diffusion = 1.0 / (3.0 * mutr)
    lam = np.sqrt(3.0 * mua * mutr)

    cumulative = np.ones_like(mua)
    if nf:
        # exp(-sum(mu*d)) equals the original product of layer attenuations.
        optical_depth = np.cumsum(mutr[:, :nf] * d[None, :], axis=1)
        cumulative[:, 1:] = np.exp(-optical_depth)
    forcing = cumulative * musp / diffusion
    # With F = forcing and the shifted source P, P'(0) = F/(lam + mutr).
    # The shifted source has a finite slope even at resonance (lam == mutr).
    p_left_prime = forcing / (lam + mutr)

    if nf:
        # Evaluate the particular source and endpoint response within each
        # finite layer; the last layer is semi-infinite.
        p_unit, dp_unit = _divided_exponential(lam[:, :nf], mutr[:, :nf], d[None, :])
        p_right = forcing[:, :nf] * p_unit
        dp_right = forcing[:, :nf] * dp_unit
        rate_left, rate_cross = _endpoint_rates(lam[:, :nf], d)

    boundary = partial_current_constant(inputs.refractive_index, inputs.outside_refractive_index)
    # Unknowns are homogeneous endpoint values [u0, v0, u1, v1, ..., u_last],
    # replacing the original decaying/growing exponential amplitudes.
    # The final semi-infinite layer has only H(z) = u_last * exp(-lam*z).
    # LAPACK band storage uses five diagonals, with row 2 as the main diagonal.
    ab = np.zeros((nw, 5, ncoef), dtype=np.float64)
    rhs = np.zeros((nw, ncoef), dtype=np.float64)

    if nf == 0:
        # A single semi-infinite layer has only the surface boundary equation.
        ab[:, 2, 0] = diffusion[:, 0] * lam[:, 0] + boundary
        rhs[:, 0] = diffusion[:, 0] * p_left_prime[:, 0]
        return BandedSystem(ab, rhs, boundary)

    # Surface condition D*phi'(0) = A*phi(0): (D*a + A)*u - D*b*v = D*P'(0).
    ab[:, 2, 0] = diffusion[:, 0] * rate_left[:, 0] + boundary
    ab[:, 1, 1] = -diffusion[:, 0] * rate_cross[:, 0]
    rhs[:, 0] = diffusion[:, 0] * p_left_prime[:, 0]

    for k in range(nf):
        # Alternate fluence-continuity and flux-continuity equations at each
        # interface, coupling adjacent layer coefficients.
        fluence_row = 2 * k + 1
        flux_row = fluence_row + 1
        left_col = 2 * k
        right_col = left_col + 1
        next_left = left_col + 2

        # Fluence continuity: v_k + P_k(d_k) = u_(k+1), since P_(k+1)(0) = 0.
        ab[:, 2, right_col] = 1.0
        ab[:, 1, next_left] = -1.0
        rhs[:, fluence_row] = -p_right[:, k]

        # Current continuity: -D_k*phi_k'(d_k) = -D_(k+1)*phi_(k+1)'(0).
        ab[:, 4, left_col] = diffusion[:, k] * rate_cross[:, k]
        ab[:, 3, right_col] = -diffusion[:, k] * rate_left[:, k]
        if k + 1 < nf:
            ab[:, 2, next_left] = -diffusion[:, k + 1] * rate_left[:, k + 1]
            ab[:, 1, next_left + 1] = diffusion[:, k + 1] * rate_cross[:, k + 1]
        else:
            ab[:, 2, next_left] = -diffusion[:, k + 1] * lam[:, k + 1]
        rhs[:, flux_row] = (
            diffusion[:, k] * dp_right[:, k] - diffusion[:, k + 1] * p_left_prime[:, k + 1]
        )
    return BandedSystem(ab, rhs, boundary)


def solve(system: BandedSystem) -> tuple[np.ndarray, np.ndarray]:
    """Return reflectance (W,) and homogeneous endpoint coefficients (W, N).

    Coefficients follow BandedSystem's ordering and differ from the exponential
    amplitudes returned internally by the original formulation.
    """
    nw, _, ncoef = system.ab.shape
    # Dense batched solves are faster for shallow systems and sufficiently
    # large batches; use LAPACK's band solver for the remaining cases.
    if ncoef <= _DENSE_SOLVE_MAX_COEFFICIENTS and nw >= _DENSE_SOLVE_MIN_BATCH:
        coefficients = _solve_shallow_batched(system.ab, system.rhs)
    else:
        coefficients = np.empty((nw, ncoef), dtype=np.float64)
        for w in range(nw):
            try:
                coefficients[w] = solve_banded(
                    (2, 2),
                    system.ab[w],
                    system.rhs[w],
                    overwrite_ab=False,
                    overwrite_b=False,
                    check_finite=False,
                )
            except LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    f"MLDT banded system is singular at wavelength index {w}"
                ) from exc
    # P(0) = 0, so the first endpoint unknown is phi(0). As in the original,
    # Rd = A*phi(0) for unit incident illumination.
    rd = system.boundary_constant * coefficients[:, 0]
    if not np.isfinite(rd).all():
        raise FloatingPointError("MLDT solve produced non-finite reflectance")
    return rd, coefficients


def _solve_shallow_batched(ab: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve band arrays (W, 5, N) against rhs (W, N), returning values (W, N).

    The chunk budget bounds only the expanded dense matrix array; output arrays
    and NumPy/LAPACK workspace require additional memory.
    """
    nw, _, ncoef = ab.shape
    rows_per_chunk = max(1, _DENSE_SOLVE_MEMORY_BYTES // (8 * ncoef * ncoef))
    coefficients = np.empty_like(rhs)
    try:
        # Expand only one memory-bounded chunk of banded matrices at a time.
        for start in range(0, nw, rows_per_chunk):
            stop = min(start + rows_per_chunk, nw)
            dense = np.zeros((stop - start, ncoef, ncoef), dtype=np.float64)
            for row in range(ncoef):
                for col in range(max(0, row - 2), min(ncoef, row + 3)):
                    dense[:, row, col] = ab[start:stop, 2 + row - col, col]
            coefficients[start:stop] = np.linalg.solve(dense, rhs[start:stop, :, None]).squeeze(-1)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("MLDT shallow batched system is singular") from exc
    return coefficients


def reflectance_validated(inputs: Inputs) -> np.ndarray:
    """Return reflectance of shape (n_wavelengths,) for validated optical inputs.

    A single semi-infinite layer uses the analytic surface equation; multiple
    layers use the equivalent assembled interface system.
    """
    if inputs.mua.shape[1] == 1:
        # Solve (D*lam + A)*u = D*P'(0) directly for a semi-infinite layer,
        # replacing the original's artificial interface between identical layers.
        mua = inputs.mua[:, 0]
        musp = inputs.musp[:, 0]
        mutr = mua + musp
        diffusion = 1.0 / (3.0 * mutr)
        lam = np.sqrt(3.0 * mua * mutr)
        forcing = musp / diffusion
        source_slope = forcing / (lam + mutr)
        boundary = partial_current_constant(
            inputs.refractive_index, inputs.outside_refractive_index
        )
        return boundary * diffusion * source_slope / (diffusion * lam + boundary)
    return solve(assemble(inputs))[0]
