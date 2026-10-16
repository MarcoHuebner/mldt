import numpy as np
import pytest

from mldt import cython


def test_cython_unavailable_is_actionable_or_matches_numpy() -> None:
    """Check native backend availability errors or numerical parity."""
    args = (np.array([[0.01]]), np.array([[1.0]]), np.array([]), 1.4)
    if not cython.is_available():
        with pytest.raises(ImportError, match="optional MLDT native extension"):
            cython.reflectance(*args)
    else:
        from mldt.numpy import reflectance

        np.testing.assert_allclose(cython.reflectance(*args), reflectance(*args))
        assert cython._core.backend_info()["implementation"] == "native-cython-lapack-banded"


def test_adjoint_matches_independent_central_difference() -> None:
    """Compare adjoint Jacobians with independent central differences."""
    if not cython.is_available():
        pytest.skip("optional native extension unavailable")
    from mldt.numpy import reflectance

    mua = np.array([[0.03, 0.05]])
    musp = np.array([[1.1, 1.7]])
    thicknesses = np.array([0.4])
    result = cython.reflectance_jacobian(mua, musp, thicknesses, 1.4)
    h = 1e-6
    plus, minus = mua.copy(), mua.copy()
    plus[0, 1] += h
    minus[0, 1] -= h
    expected = (
        reflectance(plus, musp, thicknesses, 1.4)[0] - reflectance(minus, musp, thicknesses, 1.4)[0]
    ) / (2 * h)
    assert result.d_mua[0, 1] == pytest.approx(expected, abs=2e-8)


def test_jacobian_rejects_zero_absorption() -> None:
    """Reject Jacobian requests at zero absorption."""
    if not cython.is_available():
        pytest.skip("optional native extension unavailable")
    with pytest.raises(ValueError, match="zero absorption"):
        cython.reflectance_jacobian([[0.0]], [[1.0]], [], 1.4)


def test_torch_backend_if_installed() -> None:
    """Check Torch forward values and gradients when Torch is installed."""
    torch = pytest.importorskip("torch")
    from mldt.torch import reflectance

    mua = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    musp = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    d = torch.empty(0, dtype=torch.float64, requires_grad=True)
    rd = reflectance(mua, musp, d, 1.4)
    assert rd.dtype == mua.dtype and rd.device == mua.device
    rd.backward()
    assert torch.isfinite(mua.grad).all()
    assert torch.isfinite(musp.grad).all()


def test_torch_zero_absorption_is_forward_only() -> None:
    """Allow zero-absorption forward evaluation but reject its gradient."""
    torch = pytest.importorskip("torch")
    from mldt.torch import reflectance

    mua = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    musp = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    rd = reflectance(mua, musp, torch.empty(0, dtype=torch.float64), 1.4)
    assert rd.item() == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="zero absorption"):
        rd.backward()


def test_torch_banded_solver_matches_numpy_and_differentiates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare the Torch banded solver and gradients with NumPy."""
    torch = pytest.importorskip("torch")
    import mldt.torch as torch_backend
    from mldt.numpy import reflectance

    monkeypatch.setattr(torch_backend, "_BANDED_SOLVE_MIN_WORK", 0)
    rng = np.random.default_rng(27)
    mua = rng.uniform(0.001, 0.08, (4, 20))
    musp = rng.uniform(0.3, 3.0, (4, 20))
    thicknesses = rng.uniform(0.01, 2.0, 19)
    tm = torch.tensor(mua, requires_grad=True)
    ts = torch.tensor(musp, requires_grad=True)
    td = torch.tensor(thicknesses, requires_grad=True)
    actual = torch_backend.reflectance(tm, ts, td, 1.4)
    np.testing.assert_allclose(
        actual.detach().numpy(),
        reflectance(mua, musp, thicknesses, 1.4),
        rtol=2e-12,
        atol=2e-12,
    )
    actual.sum().backward()
    assert torch.isfinite(tm.grad).all()
    assert torch.isfinite(ts.grad).all()
    assert torch.isfinite(td.grad).all()


def test_torch_banded_solver_falls_back_for_extreme_property_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the dense fallback for extreme optical-property scales."""
    torch = pytest.importorskip("torch")
    import mldt.torch as torch_backend
    from mldt.numpy import reflectance

    monkeypatch.setattr(torch_backend, "_BANDED_SOLVE_MIN_WORK", 0)
    layers = 32
    base = np.geomspace(1e-4, 30.0, layers)
    mua = np.stack((base * 0.01, base[::-1] * 0.02))
    musp = np.stack((base, base[::-1]))
    thicknesses = np.geomspace(1e-5, 1e3, layers - 1)
    tm = torch.tensor(mua, requires_grad=True)
    ts = torch.tensor(musp, requires_grad=True)
    td = torch.tensor(thicknesses, requires_grad=True)
    actual = torch_backend.reflectance(tm, ts, td, 1.7)
    np.testing.assert_allclose(
        actual.detach().numpy(),
        reflectance(mua, musp, thicknesses, 1.7),
        # This deliberately extreme system is ill-conditioned; dense LAPACK
        # solvers can differ slightly across platforms after pivoting.
        rtol=5e-8,
        atol=1e-10,
    )
    actual.sum().backward()
    assert torch.isfinite(tm.grad).all()
    assert torch.isfinite(ts.grad).all()
    assert torch.isfinite(td.grad).all()


def test_torch_reflectance_gradient_matches_central_difference() -> None:
    """Compare Torch's absorption gradient with an independent central difference."""
    torch = pytest.importorskip("torch")
    from mldt.numpy import reflectance
    from mldt.torch import reflectance as torch_reflectance

    mua = np.array([[0.03, 0.05]])
    musp = np.array([[1.1, 1.7]])
    thicknesses = np.array([0.4])
    tm = torch.tensor(mua, dtype=torch.float64, requires_grad=True)
    ts = torch.tensor(musp, dtype=torch.float64)
    td = torch.tensor(thicknesses, dtype=torch.float64)

    torch_reflectance(tm, ts, td, 1.4).sum().backward()

    h = 1e-6
    plus, minus = mua.copy(), mua.copy()
    plus[0, 1] += h
    minus[0, 1] -= h
    expected = (
        reflectance(plus, musp, thicknesses, 1.4)[0] - reflectance(minus, musp, thicknesses, 1.4)[0]
    ) / (2 * h)
    assert tm.grad[0, 1].item() == pytest.approx(expected, abs=2e-8)
