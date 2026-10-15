import numpy as np
import pytest

from mldt.original import calculateBoundaryConstant_A, getRd_MLDT, reflectance


def test_original_notebook_reference_values() -> None:
    """Check the original boundary constant against its notebook value."""
    assert calculateBoundaryConstant_A(1.4) == pytest.approx(0.16957817642653872)


@pytest.mark.parametrize(
    ("mua", "musp", "thicknesses"),
    [
        ([[0.01, 0.02], [0.03, 0.04]], [[1.0, 1.2], [1.4, 1.6]], [0.5]),
        # An empty thickness array exercises the notebook's one-layer fallback.
        ([[0.01], [0.03]], [[1.0], [1.4]], []),
    ],
    ids=["multiple-layers", "single-layer-fallback"],
)
def test_original_reflectance_matches_notebook(
    mua: list[list[float]], musp: list[list[float]], thicknesses: list[float]
) -> None:
    """Match the original reflectance wrapper to the notebook implementation."""
    np.testing.assert_array_equal(
        reflectance(mua, musp, thicknesses, 1.4),
        getRd_MLDT(mua, musp, thicknesses, 1.4),
    )


def test_original_rejects_non_air_outside_index() -> None:
    """Reject outside refractive indices unsupported by the original solver."""
    with pytest.raises(ValueError, match="only supports outside index 1.0"):
        reflectance([[0.01]], [[1.0]], [], 1.4, outside_refractive_index=1.33)
