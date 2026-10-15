import pytest

from mldt.boundary import partial_current_constant


def test_tissue_air_reference() -> None:
    """Check the partial-current constant against the tissue-air reference."""
    assert partial_current_constant(1.4, 1.0) == pytest.approx(0.1695781764181, abs=2e-13)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_index(bad: float) -> None:
    """Reject non-positive or non-finite refractive indices."""
    with pytest.raises(ValueError):
        partial_current_constant(bad)
