"""Device-native and Cython-backed PyTorch integrations.

See mldt/_system.py for the diffusion equations and endpoint formulation.
Both paths use phi = H + P with P(0) = 0 and homogeneous endpoint unknowns
[u0, v0, ..., u_last]. The final layer is semi-infinite. With depth increasing
into the medium, current is J = -D*phi'; the surface condition is
D*phi'(0) = A*phi(0), and phi and J are continuous at interfaces.
For unit incident illumination, reflectance is A*u0.

reflectance keeps optical calculations and their autograd graph on the input
device, using either dense pivoted solves or compact unpivoted elimination
with selective dense fallback. Validation and fallback decisions synchronize
tensor predicates with the host. The Fresnel constant A is calculated by the
shared cached SciPy helper on the CPU; refractive indices are scalar settings
and are not differentiated.

reflectance_cython instead bridges CPU tensors to the compiled float64 solver
and supplies a custom first-order backward from explicit adjoint Jacobians.
Both APIs support forward evaluation at zero absorption but reject backward
there, where optical-property gradients are undefined.
"""

from typing import Any

from .boundary import partial_current_constant

# Performance heuristics, not accuracy tolerances; benchmark provenance is not
# documented here, and crossover points depend on device, dtype, and batch size.
_BANDED_SOLVE_MIN_WORK = 900_000  # Minimum dense element count W*N*N for banded solving.
_BANDED_SOLVE_MIN_COEFFICIENTS = 31  # Minimum system size N for banded solving.


def _torch_module() -> Any:
    """Import PyTorch or raise an error explaining the optional dependency."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch support requires the 'torch' optional dependency") from exc
    return torch


def _validate(
    mua: Any, musp: Any, thicknesses: Any, refractive_index: float, outside: float
) -> float:
    """Validate optical tensors and return the scalar Fresnel constant A.

    mua and musp must share shape (W, L); thicknesses has shape (L - 1,).
    All tensors share a device and float32/float64 dtype. Absorption is finite
    and nonnegative; scattering and thicknesses are finite and positive.
    Scalar refractive indices are validated by the shared boundary helper.
    """
    torch = _torch_module()
    if not all(isinstance(value, torch.Tensor) for value in (mua, musp, thicknesses)):
        raise TypeError("mua, musp, and thicknesses must be torch.Tensor instances")
    if mua.ndim != 2 or musp.ndim != 2 or mua.shape != musp.shape:
        raise ValueError("mua and musp must have matching 2-D shapes")
    if mua.numel() == 0 or mua.shape[1] < 1:
        raise ValueError("mua and musp must not be empty")
    if thicknesses.ndim != 1 or thicknesses.numel() != mua.shape[1] - 1:
        raise ValueError("thicknesses must be 1-D with n_layers - 1 values")
    if mua.device != musp.device or mua.device != thicknesses.device:
        raise ValueError("all tensors must be on the same device")
    if mua.dtype != musp.dtype or mua.dtype != thicknesses.dtype:
        raise ValueError("all tensors must have the same dtype")
    if mua.dtype not in (torch.float32, torch.float64):
        raise TypeError("PyTorch MLDT supports float32 and float64")
    # Combine valid-input reductions before crossing the device/host boundary.
    # Detailed checks only run on the exceptional path.
    valid = torch.stack(
        (
            torch.isfinite(mua).all(),
            torch.isfinite(musp).all(),
            torch.isfinite(thicknesses).all(),
            (mua >= 0).all(),
            (musp > 0).all(),
            (thicknesses > 0).all(),
        )
    ).all()
    if not bool(valid):
        if not bool(torch.isfinite(mua).all()) or not bool(torch.isfinite(musp).all()):
            raise ValueError("optical properties must be finite")
        if not bool(torch.isfinite(thicknesses).all()):
            raise ValueError("thicknesses must be finite")
        if bool((mua < 0).any()):
            raise ValueError("mua must be non-negative")
        if bool((musp <= 0).any()):
            raise ValueError("musp must be strictly positive")
        raise ValueError("thicknesses must be strictly positive")
    # The scalar validation and Fresnel integration happen together.
    return partial_current_constant(refractive_index, outside)


def _source_terms(torch: Any, lam: Any, mu: Any, distance: Any) -> tuple[Any, Any]:
    """Return P/F and P'/F with the broadcast shape of lam, mu, and distance.

    lam is the diffusion decay rate, mu the transport attenuation, and distance
    local depth, normally broadcast across (W, n_finite_layers).
    P/F = (exp(-mu*z) - exp(-lam*z))/(lam**2 - mu**2).
    Shifting the original source by a homogeneous term gives P(0) = 0.
    """
    gap = torch.abs(lam - mu)
    z = distance
    az = gap * z
    # torch.where evaluates both branches: keep the regular denominator finite
    # at resonance (lam == mu), even when the series branch is selected.
    safe_gap = torch.where(gap == 0, torch.ones_like(gap), gap)
    regular = -torch.expm1(-az) / safe_gap
    # h = (1 - exp(-gap*z))/gap tends to z at resonance. The relative series
    # remainder O((gap*z)**3/24) is below each dtype's precision at its cutoff.
    series = z - gap * z.square() / 2 + gap.square() * z.pow(3) / 6
    h = torch.where(az <= 1e-4 if lam.dtype == torch.float32 else az <= 1e-7, series, regular)
    base = torch.minimum(lam, mu)
    exp_base = torch.exp(-base * z)
    value = exp_base * h / (lam + mu)
    derivative = exp_base * (torch.exp(-az) - base * h) / (lam + mu)
    return value, derivative


def _endpoint_rates(torch: Any, lam: Any, thicknesses: Any) -> tuple[Any, Any]:
    """Return a = lam*coth(lam*d), b = lam/sinh(lam*d), both shaped (W, nf).

    lam has shape (W, nf); thicknesses has shape (nf,). For u = H(0), v = H(d),
    H'(0) = -a*u + b*v and H'(d) = -b*u + a*v. Both rates tend to 1/d at lam = 0.
    """
    x = lam * thicknesses.unsqueeze(0)
    # Series through x**4 has relative remainder O(x**6), below dtype precision
    # at these cutoffs. Decaying exponentials avoid overflow in thick layers.
    cutoff = 1e-2 if lam.dtype == torch.float32 else 1e-4
    small = torch.abs(x) < cutoff
    x2 = x.square()
    a_small = (1 + x2 / 3 - x2.square() / 45) / thicknesses
    b_small = (1 - x2 / 6 + 7 * x2.square() / 360) / thicknesses
    denom = -torch.expm1(-2 * x)
    safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    decay = torch.exp(-x)
    a_regular = lam * (1 + decay.square()) / safe_denom
    b_regular = lam * 2 * decay / safe_denom
    a = torch.where(small, a_small, a_regular)
    b = torch.where(small, b_small, b_regular)
    return a, b


def _banded_solve(torch: Any, bands: Any, rhs: Any) -> tuple[Any, Any]:
    """Solve bands (W, 5, N) against rhs (W, N), returning solution and quality.

    Entry M[i, j] is stored at bands[w, 2+i-j, j], matching SciPy band storage.
    Quality (W,) is the smallest ratio of diagonal pivot magnitude to the
    largest candidate magnitude in its lower column band during elimination.
    Values below one indicate partial pivoting would choose another row.
    The caller handles fallback; this routine does not guard zero pivots.
    """
    # Split the compact five-diagonal layout into vectors for elimination.
    size = rhs.shape[1]
    zero = rhs[:, 0] * 0
    one = zero + 1
    diagonal = [bands[:, 2, index] for index in range(size)]
    upper_one = [bands[:, 1, index + 1] if index + 1 < size else zero for index in range(size)]
    upper_two = [bands[:, 0, index + 2] if index + 2 < size else zero for index in range(size)]
    lower_one = [zero] + [bands[:, 3, index - 1] for index in range(1, size)]
    lower_two = [zero, zero] + [bands[:, 4, index - 2] for index in range(2, size)]
    values = [rhs[:, index] for index in range(size)]
    pivot_quality = one

    # Replace list entries with tensor expressions rather than modifying saved
    # tensor values in place, preserving operations needed by autograd.
    # Eliminate entries below each pivot while tracking pivot stability.
    for index in range(size):
        pivot = torch.abs(diagonal[index])
        largest = pivot
        if index + 1 < size:
            largest = torch.maximum(largest, torch.abs(lower_one[index + 1]))
        if index + 2 < size:
            largest = torch.maximum(largest, torch.abs(lower_two[index + 2]))
        quality = pivot / torch.clamp_min(largest, torch.finfo(rhs.dtype).tiny)
        pivot_quality = torch.minimum(pivot_quality, quality)
        if index + 1 < size:
            factor = lower_one[index + 1] / diagonal[index]
            diagonal[index + 1] = diagonal[index + 1] - factor * upper_one[index]
            upper_one[index + 1] = upper_one[index + 1] - factor * upper_two[index]
            values[index + 1] = values[index + 1] - factor * values[index]
        if index + 2 < size:
            factor = lower_two[index + 2] / diagonal[index]
            lower_one[index + 2] = lower_one[index + 2] - factor * upper_one[index]
            diagonal[index + 2] = diagonal[index + 2] - factor * upper_two[index]
            values[index + 2] = values[index + 2] - factor * values[index]

    # Back-substitute through the resulting upper banded system.
    solution = [None] * size
    solution[-1] = values[-1] / diagonal[-1]
    if size > 1:
        solution[-2] = (values[-2] - upper_one[-2] * solution[-1]) / diagonal[-2]
    for index in range(size - 3, -1, -1):
        solution[index] = (
            values[index]
            - upper_one[index] * solution[index + 1]
            - upper_two[index] * solution[index + 2]
        ) / diagonal[index]
    return torch.stack(solution, dim=1), pivot_quality


def _dense_from_bands(torch: Any, bands: Any) -> Any:
    """Expand bands (W, 5, N) into dense matrices (W, N, N) on the same device.

    Only wavelengths selected for fallback are passed here, limiting dense
    allocation when most banded systems have acceptable pivots.
    """
    batch, _, size = bands.shape
    dense = torch.zeros((batch, size, size), dtype=bands.dtype, device=bands.device)
    for column in range(size):
        for row in range(max(0, column - 2), min(size, column + 3)):
            dense[:, row, column] = bands[:, 2 + row - column, column]
    return dense


def _guard_zero_gradient(result: Any, mua: Any) -> Any:
    """Attach a backward hook that rejects any zero absorption in this batch.

    Forward values remain available. The hook checks only when backward runs
    and passes upstream gradients through unchanged for positive absorption.
    """
    if result.requires_grad:

        def reject_zero_gradient(gradient: Any) -> Any:
            """Reject backpropagation at zero absorption."""
            if bool((mua == 0).any()):
                raise RuntimeError(
                    "MLDT backward is undefined at zero absorption; detach inputs "
                    "for a forward-only calculation"
                )
            return gradient

        result.register_hook(reject_zero_gradient)
    return result


def reflectance(
    mua: Any,
    musp: Any,
    thicknesses: Any,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> Any:
    """Return reflectance (W,) using PyTorch operations on the input device.

    mua and musp are tensors (W, L); thicknesses is a shared vector (L - 1,).
    Inputs share device and float32/float64 dtype, which the output preserves.
    Autograd tracks mua, musp, and thicknesses for positive absorption;
    refractive indices are nondifferentiable scalar configuration values.
    The last layer extends to infinite depth. Invalid inputs raise ValueError
    or TypeError; failed pivoted solves propagate PyTorch's solver exception.
    """
    # Validate tensors, then derive the optical coefficients shared by layers.
    torch = _torch_module()
    boundary = _validate(mua, musp, thicknesses, refractive_index, outside_refractive_index)
    nw, nl = mua.shape
    nf = nl - 1
    ncoef = 2 * nf + 1
    mu = mua + musp
    diffusion = 1 / (3 * mu)
    lam = torch.sqrt(3 * mua * mu)
    # Propagate the incident source through each finite layer.
    cumulative = torch.ones_like(mua)
    if nf:
        cumulative = torch.cat(
            (
                cumulative[:, :1],
                torch.exp(-torch.cumsum(mu[:, :nf] * thicknesses.unsqueeze(0), dim=1)),
            ),
            dim=1,
        )
    forcing = cumulative * musp / diffusion
    # P'(0) = F/(lam + mu) is finite even at source resonance (lam == mu).
    dp_left = forcing / (lam + mu)
    a_bc = torch.as_tensor(boundary, dtype=mua.dtype, device=mua.device)
    if nf == 0:
        # Solve (D*lam + A)*u = D*P'(0) directly, including zero absorption.
        result = a_bc * diffusion[:, 0] * dp_left[:, 0] / (diffusion[:, 0] * lam[:, 0] + a_bc)
        return _guard_zero_gradient(result, mua)

    # Evaluate source terms and endpoint rates for each finite layer.
    p_unit, dp_unit = _source_terms(torch, lam[:, :nf], mu[:, :nf], thicknesses.unsqueeze(0))
    p_right = forcing[:, :nf] * p_unit
    dp_right = forcing[:, :nf] * dp_unit
    rate_left, rate_cross = _endpoint_rates(torch, lam[:, :nf], thicknesses)
    rhs = torch.zeros((nw, ncoef), dtype=mua.dtype, device=mua.device)
    # Choose a solve layout: dense LU wins for tiny batches; banded
    # elimination wins when the quadratic dense storage/work becomes material.
    use_banded = (
        ncoef >= _BANDED_SOLVE_MIN_COEFFICIENTS and nw * ncoef * ncoef >= _BANDED_SOLVE_MIN_WORK
    )
    # Unknown ordering is [u0, v0, ..., u_last] in either layout.
    # Surface condition: (D*a + A)*u - D*b*v = D*P'(0).
    if use_banded:
        matrix = torch.zeros((nw, 5, ncoef), dtype=mua.dtype, device=mua.device)
        matrix[:, 2, 0] = diffusion[:, 0] * rate_left[:, 0] + a_bc
        matrix[:, 1, 1] = -diffusion[:, 0] * rate_cross[:, 0]
    else:
        matrix = torch.zeros((nw, ncoef, ncoef), dtype=mua.dtype, device=mua.device)
        matrix[:, 0, 0] = diffusion[:, 0] * rate_left[:, 0] + a_bc
        matrix[:, 0, 1] = -diffusion[:, 0] * rate_cross[:, 0]
    rhs[:, 0] = diffusion[:, 0] * dp_left[:, 0]
    for k in range(nf):
        # Row r1: v_k + P_k(d_k) = u_(k+1) (fluence continuity).
        # Row r2: -D_k*phi_k'(d_k) = -D_(k+1)*phi_(k+1)'(0) (current continuity).
        # The last layer contributes H'(0) = -lam*u with no growing mode.
        r1, r2, c = 2 * k + 1, 2 * k + 2, 2 * k
        if use_banded:
            matrix[:, 2, c + 1] = 1
            matrix[:, 1, c + 2] = -1
            matrix[:, 4, c] = diffusion[:, k] * rate_cross[:, k]
            matrix[:, 3, c + 1] = -diffusion[:, k] * rate_left[:, k]
            if k + 1 < nf:
                matrix[:, 2, c + 2] = -diffusion[:, k + 1] * rate_left[:, k + 1]
                matrix[:, 1, c + 3] = diffusion[:, k + 1] * rate_cross[:, k + 1]
            else:
                matrix[:, 2, c + 2] = -diffusion[:, k + 1] * lam[:, k + 1]
        else:
            matrix[:, r1, c + 1] = 1
            matrix[:, r1, c + 2] = -1
            matrix[:, r2, c] = diffusion[:, k] * rate_cross[:, k]
            matrix[:, r2, c + 1] = -diffusion[:, k] * rate_left[:, k]
            if k + 1 < nf:
                matrix[:, r2, c + 2] = -diffusion[:, k + 1] * rate_left[:, k + 1]
                matrix[:, r2, c + 3] = diffusion[:, k + 1] * rate_cross[:, k + 1]
            else:
                matrix[:, r2, c + 2] = -diffusion[:, k + 1] * lam[:, k + 1]
        rhs[:, r1] = -p_right[:, k]
        rhs[:, r2] = diffusion[:, k] * dp_right[:, k] - diffusion[:, k + 1] * dp_left[:, k + 1]
    # Solve the selected layout, retrying unstable banded rows with pivoted LU.
    if use_banded:
        coefficients, pivot_quality = _banded_solve(torch, matrix, rhs)
        # Match the decision made by partial pivoting: if a lower band entry
        # exceeds the diagonal pivot, route that row to pivoted dense LU.
        property_scale = torch.amax(mu, dim=1) / torch.amin(mu, dim=1)
        # Large transport contrast also triggers fallback as a conditioning
        # heuristic; this ratio is not a matrix condition-number estimate.
        unstable = (pivot_quality < 1) | (property_scale > 1e5)
        if bool(unstable.any()):
            unstable_matrix = _dense_from_bands(torch, matrix[unstable])
            fallback = torch.linalg.solve(unstable_matrix, rhs[unstable].unsqueeze(-1)).squeeze(-1)
            coefficients = coefficients.clone()
            coefficients[unstable] = fallback
    else:
        coefficients = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    # P(0) = 0 makes the first endpoint unknown surface fluence: Rd = A*u0.
    return _guard_zero_gradient(a_bc * coefficients[:, 0], mua)


def reflectance_cython(
    mua: Any,
    musp: Any,
    thicknesses: Any,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> Any:
    """Return CPU reflectance (W,) with a custom first-order autograd bridge.

    Input shapes and physical domains match reflectance. All inputs must be
    CPU tensors sharing float32/float64 dtype; output preserves that dtype.
    Forward and explicit Jacobian calculations use detached float64 NumPy data.
    Backward supports mua, musp, and thicknesses at positive absorption, but
    once_differentiable disables higher-order gradients. The optional compiled
    backend must be installed; refractive indices are fixed scalar settings.
    """
    # The native extension accepts CPU NumPy arrays, so reject other devices.
    torch = _torch_module()
    if any(t.device.type != "cpu" for t in (mua, musp, thicknesses)):
        raise ValueError("the Cython-backed Torch implementation is CPU-only")
    boundary = _validate(mua, musp, thicknesses, refractive_index, outside_refractive_index)
    from . import cython as cython_backend

    class _CythonReflectance(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, m: Any, s: Any, d: Any) -> Any:
            """Run the native reflectance calculation and save inputs."""
            # Save original tensors for backward and pass detached float64 data
            # to the compiled forward kernel.
            ctx.save_for_backward(m, s, d)
            value = cython_backend.reflectance(
                m.detach().double().numpy(),
                s.detach().double().numpy(),
                d.detach().double().numpy(),
                refractive_index,
                outside_refractive_index=outside_refractive_index,
            )
            return torch.as_tensor(value, dtype=m.dtype, device=m.device)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, Any]:
            """Apply the adjoint Jacobian to the incoming gradient."""
            # Optical Jacobians have shape (W, L); the thickness Jacobian has
            # shape (W, L-1). Multiply by upstream weights (W,) and sum over
            # wavelengths for thicknesses, which are shared by the whole batch.
            m, s, d = ctx.saved_tensors
            if bool((m == 0).any()):
                raise RuntimeError("MLDT backward is undefined at zero absorption")
            jac = cython_backend.reflectance_jacobian(
                m.detach().double().numpy(),
                s.detach().double().numpy(),
                d.detach().double().numpy(),
                refractive_index,
                outside_refractive_index=outside_refractive_index,
            )
            gm = torch.as_tensor(jac.d_mua, dtype=m.dtype) * grad_output[:, None]
            gs = torch.as_tensor(jac.d_musp, dtype=s.dtype) * grad_output[:, None]
            gd = (torch.as_tensor(jac.d_thickness, dtype=d.dtype) * grad_output[:, None]).sum(0)
            return gm, gs, gd

    del boundary
    return _CythonReflectance.apply(mua, musp, thicknesses)
