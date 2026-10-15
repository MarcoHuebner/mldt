# Unified accuracy and performance evaluation

Run commands from the repository root. For smoke and standard runs, install the
locked development and acceleration environment:

```bash
uv sync --locked --python 3.12 --extra accelerated --extra dev
```

For the publication profile with the mcmlnet neural surrogate, include all
extras in the same synchronization command:

```bash
uv sync --locked --python 3.12 \
  --extra accelerated --extra dev --extra mcmlnet
```

Git and network access are required the first time because mcmlnet is pinned to
a Git commit. A C compiler is required for the native backend. The model weights
are separate files and are not installed by the `mcmlnet` extra. See the
repository [README](README.md#installation) for the dependency and CUDA checks.

The suite evaluates the original notebook port, current NumPy/SciPy, Cython,
PyTorch CPU, and PyTorch GPU through one runner. Each stage runs in an isolated
subprocess. Missing optional backends are recorded without preventing other
backends or stages from completing.

## Standard commands

Use the reduced deterministic profile for CI and development checks:

```bash
uv run python scripts/evaluations/run_suite.py \
  --profile smoke --output-dir benchmark-results/smoke
```

The standard profile includes equivalence, robustness, performance,
representative inference, high-layer, and gradient stages:

```bash
uv run python scripts/evaluations/run_suite.py \
  --profile standard --output-dir benchmark-results/standard
```

The publication profile increases case counts and repetitions and adds external
MCX/MCML evaluation:

```bash
uv run python scripts/evaluations/run_suite.py \
  --profile publication --output-dir benchmark-results/publication
```

The external stage also evaluates mcmlnet when `--model-dir`,
`--model-version`, and `--checkpoint-path` are supplied. These may instead be
set as `model_dir`, `model_version`, and `checkpoint_path` in a repository-root
`.env` file; see `.env.example`. `model_dir` is the directory containing model
version folders. The selected `<model_dir>/<model_version>/` directory must
contain `config_log.yaml`; `checkpoint_path` is interpreted relative to that
directory.

Validate the surrogate separately before a long publication run:

```bash
uv run python -c "import mcmlnet, hydra, omegaconf, shap, numba, torchvision; print('mcmlnet imports: ok')"
uv run python scripts/evaluations/run_suite.py \
  --profile smoke --stages external --backends numpy \
  --output-dir benchmark-results/external-smoke
```

The external stage is intentionally non-fatal when model configuration or
checkpoint loading fails. Check `stages/external.json` at
`mcmlnet.status`; a successful surrogate evaluation reports `"ok"`.

## Selection and resuming

Select stages and backends with comma-separated values:

```bash
uv run python scripts/evaluations/run_suite.py \
  --profile standard \
  --stages equivalence,robustness,inference \
  --backends original,numpy,cython,torch-cpu,torch-gpu \
  --output-dir benchmark-results/selected
```

Available stages are `equivalence`, `robustness`, `performance`, `inference`,
`high-layer`, `gradients`, `external`, and `profile`. `--resume` reuses a stage
only when its complete configuration hash matches the current request.
`--dtype float32` selects application/inference precision;
`--dtypes float32,float64` controls the performance matrix. Float64 remains the
correctness default.

Backend selection applies to stage execution and reporting. `equivalence`
requires `original`; `robustness` and `gradients` require `numpy` as their
stable numerical reference. Non-Torch backends do not provide native float32,
so those rows are recorded as unsupported when float32 is requested.

Profiling is intentionally explicit:

```bash
uv run python scripts/evaluations/run_suite.py \
  --profile standard --stages profile \
  --output-dir benchmark-results/profile
```

Python profiles expose CPU work and GPU launch/dispatch overhead. Use a device
profiler for CUDA kernel analysis.

## Timing definitions

- **Solver-only:** arrays or tensors are prepared before timing, with CUDA
  synchronized around the forward call.
- **End-to-end:** conversion, allocation, device transfer, synchronized forward
  execution, and host result transfer are timed.
- Every result retains raw timing samples plus mean, standard deviation,
  median, and interquartile values.

The inference stage uses a three-layer model over 10,000 spectra with 100
wavelengths each. It compares one large call with 1,000 calls of ten complete
spectra for every available primary backend. Its timings are solver-only: all
path-specific batches are prepared before the measured pass. The smoke profile
uses 20 spectra and five wavelengths to keep CI fast.

## Outputs and schema

Each output directory contains:

- `results.json`: authoritative consolidated, versioned result document;
- `stages/*.json`: independently reusable stage documents;
- `configs/*.json`: exact stage configurations;
- `report.md`: methodology, availability, status, interpretation, and figures;
- focused PNG figures for every completed applicable stage;
- `profiles/` when profiling is selected.

The possible figures are `equivalence.png`, `robustness.png`,
`performance.png`, `inference.png`, `high-layer.png`, and `gradients.png`.
The external stage creates four 3x3 figures: `external-absolute.png`,
`external-difference.png`, `external-absolute-specular.png`, and
`external-difference-specular.png`. The specular variants add the
normal-incidence Fresnel reflectance to MCML and mcmlnet. Only figures for
selected, successfully completed stages are created.

Regenerate analysis without rerunning benchmarks:

```bash
uv run python scripts/evaluations/analyze_suite.py \
  benchmark-results/standard/results.json \
  --output-dir benchmark-results/standard
```

The schema records configuration hashes, seeds, stage settings, dependency and
hardware information, Git state, backend availability, failures, raw timings,
accuracy summaries, and worst-case indices. Incompatible schema versions are
rejected.

## Interpretation and reproducibility

The equivalence stage uses the original notebook port on regular,
well-conditioned inputs. The robustness stage uses stable NumPy/SciPy and
analytic limits as references. Differences from the original at resonance,
zero absorption, and optically thick limits are intentional corrections.
Neither the report nor the suite claims bitwise identity between backends.

For publication measurements, keep the machine idle and use fixed CPU/GPU
power modes. Preserve JSON with the figures. GPU results require a compatible
PyTorch build and driver; Cython results require the native extension.
Unavailable implementations remain visible in JSON and the report.

The external stage reads provenance-tagged MCX and optional MCML references in
`data/`; external physical-model discrepancies are reported separately from
cross-backend numerical errors. Regenerate the local MCML reference on a CUDA
machine with the retained utility (and installed `mcml` executable):

```bash
uv run python scripts/evaluations/resimulate_mcml_reference.py
```
