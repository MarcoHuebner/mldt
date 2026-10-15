"""Edge cases for the Torch backend's shared input validation."""

import pytest

from mldt.boundary import partial_current_constant
from mldt.torch import _validate


def _inputs(torch):
    """Return valid two-layer inputs with multiple samples."""
    return [
        torch.full((2, 2), 0.01, dtype=torch.float64),
        torch.ones((2, 2), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    ]


@pytest.mark.parametrize("index", range(3), ids=["mua", "musp", "thicknesses"])
def test_requires_tensors(index):
    """Reject optical inputs supplied as lists instead of tensors."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[index] = inputs[index].tolist()
    with pytest.raises(TypeError, match="must be torch.Tensor instances"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize(
    "index,shape",
    [
        (0, ()),
        (0, (2,)),
        (0, (1, 2, 2)),
        (1, ()),
        (1, (2,)),
        (1, (1, 2, 2)),
        (0, (1, 2)),
        (1, (2, 3)),
    ],
)
def test_requires_matching_two_dimensional_properties(index, shape):
    """Reject optical tensors with incompatible dimensions or shapes."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[index] = torch.ones(shape, dtype=torch.float64)
    with pytest.raises(ValueError, match="matching 2-D shapes"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (0, 0)])
def test_rejects_empty_properties(shape):
    """Reject optical tensors with empty wavelength or layer axes."""
    torch = pytest.importorskip("torch")
    inputs = [torch.empty(shape, dtype=torch.float64) for _ in range(2)]
    with pytest.raises(ValueError, match="must not be empty"):
        _validate(*inputs, torch.empty(0, dtype=torch.float64), 1.4, 1.0)


@pytest.mark.parametrize("shape", [(), (1, 1), (0,), (2,)])
def test_requires_one_thickness_per_finite_layer(shape):
    """Require a thickness vector with one entry per finite layer."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[2] = torch.ones(shape, dtype=torch.float64)
    with pytest.raises(ValueError, match="thicknesses must be 1-D with n_layers - 1 values"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("index", [1, 2], ids=["musp", "thicknesses"])
def test_rejects_mixed_devices(index):
    """Reject input tensors placed on different devices."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    # Meta tensors exercise device validation without requiring a GPU.
    inputs[index] = inputs[index].to("meta")
    with pytest.raises(ValueError, match="all tensors must be on the same device"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("index", range(3), ids=["mua", "musp", "thicknesses"])
def test_rejects_mixed_dtypes(index):
    """Reject input tensors with different numeric dtypes."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[index] = inputs[index].float()
    with pytest.raises(ValueError, match="all tensors must have the same dtype"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int64", "bool", "complex64"])
def test_rejects_unsupported_dtypes(dtype):
    """Reject tensor dtypes other than float32 and float64."""
    torch = pytest.importorskip("torch")
    inputs = [value.to(getattr(torch, dtype)) for value in _inputs(torch)]
    with pytest.raises(TypeError, match="supports float32 and float64"):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("index", range(3), ids=["mua", "musp", "thicknesses"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_values(index, value):
    """Reject NaN and infinite values in any optical input tensor."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[index].view(-1)[-1] = value
    message = "thicknesses must be finite" if index == 2 else "optical properties must be finite"
    with pytest.raises(ValueError, match=message):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize(
    "index,value,message",
    [
        (0, -1e-12, "mua must be non-negative"),
        (1, 0.0, "musp must be strictly positive"),
        (1, -1e-12, "musp must be strictly positive"),
        (2, 0.0, "thicknesses must be strictly positive"),
        (2, -1e-12, "thicknesses must be strictly positive"),
    ],
)
def test_rejects_out_of_range_values(index, value, message):
    """Reject negative absorption and nonpositive scattering or thicknesses."""
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs[index].view(-1)[-1] = value
    with pytest.raises(ValueError, match=message):
        _validate(*inputs, 1.4, 1.0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("layers", [1, 2])
def test_accepts_zero_absorption_and_positive_boundary_values(dtype, layers):
    """Accept zero absorption and tiny positive scattering and thicknesses."""
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype)
    mua = torch.zeros((1, layers), dtype=dtype)
    musp = torch.full_like(mua, torch.finfo(dtype).tiny)
    thicknesses = torch.full((layers - 1,), torch.finfo(dtype).tiny, dtype=dtype)
    assert _validate(mua, musp, thicknesses, 1.4, 1.0) == partial_current_constant(1.4, 1.0)


@pytest.mark.parametrize("index,name", [(0, "n_tissue"), (1, "n_outside")])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_validates_refractive_indices(index, name, value):
    """Reject nonpositive or nonfinite tissue and outside refractive indices."""
    torch = pytest.importorskip("torch")
    indices = [1.4, 1.0]
    indices[index] = value
    with pytest.raises(ValueError, match=f"{name} must be a positive finite scalar"):
        _validate(*_inputs(torch), *indices)
