from typing import Any

import numpy as np
import pytest

from mldt._system import BandedSystem, assemble, solve
from mldt._validation import validate_numpy
from mldt.boundary import partial_current_constant
from mldt.numpy import reflectance


def analytic_one_layer(mua: float, musp: float, nr: float = 1.4) -> float:
    """Compute the analytic one-layer reflectance reference."""
    mu = mua + musp
    diffusion = 1.0 / (3.0 * mu)
    lam = np.sqrt(3.0 * mua * mu)
    forcing = musp / diffusion
    source_slope = forcing / (lam + mu)
    a = partial_current_constant(nr)
    return a * diffusion * source_slope / (diffusion * lam + a)


@pytest.mark.parametrize(
    "n_wavelengths,message",
    [
        (2, "MLDT banded system is singular at wavelength index 1"),
        (16, "MLDT shallow batched system is singular"),
    ],
    ids=["banded", "batched-dense"],
)
def test_singular_system_reports_solver_context(n_wavelengths: int, message: str) -> None:
    """Trigger real singular solves in both paths and preserve the underlying error."""
    # Identity matrices are solvable until the second wavelength's last row
    # becomes zero. Construct the algebraic edge case directly, independently
    # of whether physically valid optical inputs can produce a singular system.
    ab = np.zeros((n_wavelengths, 5, 3))
    ab[:, 2, :] = 1.0
    ab[1, 2, -1] = 0.0
    system = BandedSystem(ab, np.ones((n_wavelengths, 3)), boundary_constant=0.17)

    with pytest.raises(np.linalg.LinAlgError, match=message) as error:
        solve(system)

    assert isinstance(error.value.__cause__, np.linalg.LinAlgError)


@pytest.mark.parametrize("mua,musp", [(0.01, 1.0), (1.0, 2.0), (0.0, 2.0)])
def test_true_one_layer_matches_analytic_limit(mua: float, musp: float) -> None:
    """Compare the one-layer backend result with its analytic limit."""
    actual = reflectance([[mua]], [[musp]], [], 1.4)[0]
    assert actual == pytest.approx(analytic_one_layer(mua, musp), rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("wavelength_count", [1, 16], ids=["banded", "batched-dense"])
@pytest.mark.parametrize("mua,musp", [(0.01, 1.0), (1.0, 2.0), (0.0, 2.0)])
def test_single_layer_assembly_without_finite_layers(
    wavelength_count: int, mua: float, musp: float
) -> None:
    """Exercise the single surface equation without the reflectance shortcut."""
    inputs = validate_numpy(
        np.full((wavelength_count, 1), mua),
        np.full((wavelength_count, 1), musp),
        [],
        1.4,
    )
    system = assemble(inputs)

    diffusion = 1.0 / (3.0 * (mua + musp))
    lam = np.sqrt(3.0 * mua * (mua + musp))
    boundary = partial_current_constant(1.4)
    expected_ab = np.zeros((wavelength_count, 5, 1))
    expected_ab[:, 2, 0] = diffusion * lam + boundary
    np.testing.assert_allclose(system.ab, expected_ab, rtol=2e-13, atol=0.0)
    assert system.rhs.shape == (wavelength_count, 1)
    np.testing.assert_allclose(system.rhs, musp / (lam + mua + musp), rtol=2e-13)
    assert system.boundary_constant == boundary

    rd, coefficients = solve(system)
    expected_rd = analytic_one_layer(mua, musp)
    assert coefficients.shape == (wavelength_count, 1)
    np.testing.assert_allclose(rd, expected_rd, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(coefficients[:, 0], expected_rd / boundary, rtol=2e-13)


def test_resonance_is_continuous() -> None:
    """Check numerical continuity at the source resonance."""
    values = [reflectance([[1.0]], [[2.0 + offset]], [], 1.4)[0] for offset in (-1e-7, 0.0, 1e-7)]
    assert values[1] == pytest.approx(0.1123976, abs=1e-7)
    assert max(values) - min(values) < 2e-8


def test_optically_thick_layer_remains_finite() -> None:
    """Check that an optically thick layer yields finite reflectance."""
    rd = reflectance([[0.01, 0.01]], [[10.0, 10.0]], [100_000.0], 1.4)
    assert np.isfinite(rd).all()
    assert 0.0 < rd[0] < 1.0


def test_zero_absorption_homogeneous_multilayer_limit() -> None:
    """Check the homogeneous multilayer limit at zero absorption."""
    rd = reflectance([[0.0, 0.0, 0.0]], [[2.0, 2.0, 2.0]], [0.1, 1000.0], 1.4)
    assert rd[0] == pytest.approx(1.0, abs=2e-13)


def test_banded_solution_residual() -> None:
    """Check the computed coefficients satisfy the banded system."""
    inputs = validate_numpy([[0.01, 0.03, 0.02]], [[1.0, 1.5, 0.8]], [0.02, 4.0], 1.4)
    system = assemble(inputs)
    _, coefficients = solve(system)
    n = coefficients.shape[1]
    dense = np.zeros((n, n))
    for row in range(n):
        for col in range(max(0, row - 2), min(n, row + 3)):
            dense[row, col] = system.ab[0, 2 + row - col, col]
    residual = dense @ coefficients[0] - system.rhs[0]
    assert np.linalg.norm(residual, ord=np.inf) < 2e-13


def test_shallow_batched_solve_matches_individual_solves() -> None:
    """Compare batched shallow solves with separate wavelength solves."""
    rng = np.random.default_rng(18)
    mua = rng.uniform(0.001, 0.1, (32, 8))
    musp = rng.uniform(0.2, 4.0, (32, 8))
    thicknesses = rng.uniform(0.01, 3.0, 7)
    batched = reflectance(mua, musp, thicknesses, 1.4)
    individual = np.array(
        [
            reflectance(mua[index : index + 1], musp[index : index + 1], thicknesses, 1.4)[0]
            for index in range(32)
        ]
    )
    np.testing.assert_allclose(batched, individual, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    "mua,musp,thicknesses",
    [
        ([[-1.0]], [[1.0]], []),
        ([[0.1]], [[0.0]], []),
        ([[np.nan]], [[1.0]], []),
        ([[0.1, 0.2]], [[1.0, 1.0]], []),
        ([[0.1, 0.2]], [[1.0, 1.0]], [0.0]),
    ],
)
def test_invalid_inputs_raise(mua: Any, musp: Any, thicknesses: Any) -> None:
    """Reject invalid optical arrays and layer thicknesses."""
    with pytest.raises(ValueError):
        reflectance(mua, musp, thicknesses, 1.4)


@pytest.mark.parametrize(
    "mua,musp,thicknesses,refractive_index,outside_refractive_index,message",
    [
        ([], [], [], 1.4, 1.0, "mua and musp must be 2-D"),
        ([[0.1]], [[1.0, 2.0]], [], 1.4, 1.0, "does not match"),
        (np.empty((0, 1)), np.empty((0, 1)), [], 1.4, 1.0, "at least one wavelength"),
        (np.empty((1, 0)), np.empty((1, 0)), [], 1.4, 1.0, "at least one wavelength"),
        ([[0.1, 0.2]], [[1.0, 2.0]], [[0.5]], 1.4, 1.0, "must be a 1-D array"),
        ([[0.1, 0.2]], [[1.0, 2.0]], [], 1.4, 1.0, "must contain 1 values"),
        ([[0.1]], [[1.0]], [], 0.0, 1.0, "refractive_index must be"),
        ([[0.1]], [[1.0]], [], 1.4, np.inf, "outside_refractive_index must be"),
    ],
    ids=[
        "non-2d-optical-arrays",
        "mismatched-optical-shapes",
        "empty-wavelength-axis",
        "empty-layer-axis",
        "non-1d-thicknesses",
        "wrong-thickness-count",
        "invalid-tissue-index",
        "nonfinite-outside-index",
    ],
)
def test_invalid_input_structure_and_indices_raise(
    mua: Any,
    musp: Any,
    thicknesses: Any,
    refractive_index: float,
    outside_refractive_index: float,
    message: str,
) -> None:
    """Reject invalid input shapes, thickness counts, and refractive indices."""
    with pytest.raises(ValueError, match=message):
        reflectance(
            mua,
            musp,
            thicknesses,
            refractive_index,
            outside_refractive_index=outside_refractive_index,
        )
