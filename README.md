# MLDT: Multi-Layer Diffusion Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**MLDT** is a fast, open-source Python package for calculating steady-state diffuse reflectance spectra ($R_d$) in multi-layered turbid media using the $P_1$ diffusion approximation.

By formulating multi-layer boundary conditions as a banded linear system, MLDT
supports efficient batched forward calculations for iterative optical-property
inference in tissue spectroscopy and biomedical imaging. Actual latency depends
on the backend, problem shape, precision, and hardware; use the unified
evaluation suite below for reproducible measurements.

The maintained solvers are provided by the `mldt` package. The historical
notebook implementation is retained as a publication artifact and as the
regular-domain equivalence baseline; it is not the reference for singular or
limiting cases fixed by the maintained implementations.

## Backends and gradient support

| Backend | Method | Devices | Gradient support |
| --- | --- | --- | --- |
| NumPy/SciPy | `mldt.numpy.reflectance` | CPU | Forward only; no built-in gradient API |
| Native Cython/LAPACK | `mldt.cython.reflectance` | CPU | Forward only; use `mldt.cython.reflectance_jacobian` for explicit Jacobians |
| PyTorch | `mldt.torch.reflectance` | CPU or CUDA | PyTorch autograd for `mua`, `musp`, and `thicknesses` |
| PyTorch + Cython | `mldt.torch.reflectance_cython` | CPU | First-order PyTorch autograd for `mua`, `musp`, and `thicknesses` |

`mldt.cython.reflectance_jacobian` returns the reflectance together with
Jacobians `d_mua`, `d_musp`, and `d_thickness`. The Cython-backed PyTorch
method uses these Jacobians in its backward pass and is once-differentiable,
so it does not support higher-order gradients. Refractive indices are scalar
configuration values and are not differentiated by either gradient-capable
method. Optical-property gradients are undefined when any absorption
coefficient (`mua`) is zero; forward evaluation at zero absorption remains
supported.

## Installation

Install [uv](https://docs.astral.sh/uv/) and Git first. A C compiler is needed
to build the optional native extension. Python 3.12 is the tested choice for
the mcmlnet compatibility stack; uv can install it when necessary:

```bash
uv python install 3.12
```

For the maintained CPU solver alone:

```bash
uv sync --locked --python 3.12
```

For development and all MLDT solver backends:

```bash
uv sync --locked --python 3.12 --extra accelerated --extra dev
```

For the complete publication environment, including the mcmlnet neural
surrogate, use all three extras together:

```bash
uv sync --locked --python 3.12 \
  --extra accelerated --extra dev --extra mcmlnet
```

Do not follow the full installation with `uv sync --extra mcmlnet` alone: uv
synchronizes the environment to exactly the selected extras and would remove
Cython, Matplotlib, Ruff, and other development packages. The `mcmlnet` extra
pins the Git revision and its compatible SHAP, Numba, llvmlite, and NumPy
versions. Its Git dependency therefore requires network access during the first
installation.

The trained mcmlnet checkpoint is not distributed by this repository or by the
Python extra. Copy `.env.example` to `.env` and set `model_dir`,
`model_version`, and `checkpoint_path` to the separately obtained model files.
The selected version directory must contain `config_log.yaml` and the configured
checkpoint path.

Check the installed solver and surrogate paths with:

```bash
uv run python -c "import torch; from mldt import cython; print('native:', cython.is_available()); print('GPU:', torch.cuda.is_available())"
uv run python -c "import mcmlnet, hydra, omegaconf, shap, numba, torchvision; print('mcmlnet imports: ok')"
```

The import check validates Python dependencies, not the checkpoint. Run the
`external` stage and inspect its `mcmlnet.status` result to validate checkpoint
loading. CUDA benchmarking additionally requires a compatible NVIDIA driver and
a CUDA-enabled PyTorch build; `torch.cuda.is_available()` must print `True`.

### Rebuilding the native extension

Package builds run Cython automatically from
`mldt/_native/core.pyx`, then compile the generated C extension. To regenerate
`core.c` and build the extension in place manually, run:

```bash
uv run --extra native cython -3 mldt/_native/core.pyx -o mldt/_native/core.c
uv run --extra native python setup.py build_ext --inplace
```

The generated `core.c` is kept alongside the Cython source, but `core.pyx` is
the source to edit. `MLDT_DISABLE_NATIVE=1` skips the native extension during a
package build.

## Usage

Pass `mua` and `musp` as 2-D arrays shaped `(n_wavelengths, n_layers)`. The
1-D `thicknesses` array contains one value per boundary between layers, so a
two-layer model has one thickness. The NumPy solver returns one reflectance
value per wavelength:

```python
from mldt.numpy import reflectance

mua = [[0.01, 0.02], [0.015, 0.025], [0.02, 0.03]]  # 3 wavelengths, 2 layers
musp = [[1.0, 1.2], [1.1, 1.3], [1.2, 1.4]]       # same shape as mua
thicknesses = [0.1]                                  # one interface, in cm

rd = reflectance(mua, musp, thicknesses, refractive_index=1.4)
print(rd)  # NumPy array with shape (3,)
```

Run formatting, linting, and tests with:

```bash
uv run ruff format mldt tests scripts setup.py
uv run ruff check mldt tests scripts setup.py
uv run pytest
```

See [benchmarks.md](benchmarks.md) for the single runner and analysis commands
covering regular-domain equivalence, edge-case robustness, CPU/GPU performance,
the representative inference workload, 101-layer cases, gradients, profiling,
and optional MCX/MCML/mcmlnet comparisons.

## Citation
If you use MLDT in your research, please cite our manuscript published in the Journal of Biomedical Optics (JBO):

Thomas T. Livecchi, Mark C. Pierce, Steven L. Jacques, MLDT An open-source multi-layer diffusion theory Python implementation, J. Biomed. Opt. 31(9), 095006 (2026), doi: 10.1117/1.JBO.31.9.095006.
