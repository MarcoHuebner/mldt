"""Implementations of the unified evaluation stages."""

from __future__ import annotations

import cProfile
import json
import pstats
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backends import Backend, discover, evaluate
from .cases import REGIMES, high_layer_case, ordinary_case, regime_case, spectral_batches
from .metrics import error_summary, measure


def _safe_evaluate(backend: Backend, inputs: Any, dtype: str) -> tuple[np.ndarray | None, str]:
    """Evaluate a backend while converting warnings and exceptions to status text."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = evaluate(backend, inputs, dtype=dtype)
        status = "; ".join(str(item.message) for item in caught) or "ok"
        if not np.all(np.isfinite(value)):
            status = "non-finite result"
        return value, status
    except Exception as exc:  # noqa: BLE001 - stages record per-backend failures
        return None, f"{type(exc).__name__}: {exc}"


def _setup(config: dict[str, Any]) -> tuple[dict[str, Backend], dict[str, str]]:
    """Discover the backends requested by a stage configuration."""
    return discover(config["backends"])


def _finite_or_label(value: float) -> float | str:
    """Convert non-finite values to JSON-safe labels."""
    value = float(value)
    if np.isfinite(value):
        return value
    return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def _errors_with_worst(
    actual: Any,
    reference: Any,
    inputs: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    """Summarize numerical errors and include details for the worst input."""
    summary = error_summary(actual, reference, atol=atol, rtol=rtol)
    index = summary["worst_index"]
    if index is not None:
        summary["worst_case"] = {
            "mua": np.asarray(inputs[0])[index].tolist(),
            "musp": np.asarray(inputs[1])[index].tolist(),
            "thicknesses": np.asarray(inputs[2]).tolist(),
            "refractive_index": inputs[3],
            "outside_refractive_index": inputs[4],
            "reference": _finite_or_label(np.asarray(reference).reshape(-1)[index]),
            "actual": _finite_or_label(np.asarray(actual).reshape(-1)[index]),
        }
    return summary


def equivalence(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Compare all requested backends with the original on regular inputs."""
    backends, unavailable = _setup(config)
    if "original" not in backends:
        raise RuntimeError("equivalence requires the original backend")
    rng = np.random.default_rng(config["seed"])
    # Use fixed seeded workloads so backend comparisons are repeatable.
    workloads = config["settings"]["equivalence_workloads"]
    rows = []
    for count, layers in workloads:
        inputs = ordinary_case(rng, count, layers)
        reference = evaluate(backends["original"], inputs, dtype="float64")
        for name in config["backends"]:
            if name not in backends:
                rows.append(
                    {"workload": f"{count}x{layers}", "backend": name, "status": unavailable[name]}
                )
                continue
            value, status = _safe_evaluate(backends[name], inputs, "float64")
            row = {
                "workload": f"{count}x{layers}",
                "rows": count,
                "layers": layers,
                "backend": name,
                "status": status,
            }
            if value is not None:
                row["errors"] = _errors_with_worst(value, reference, inputs)
                split = np.concatenate(
                    [
                        evaluate(
                            backends[name],
                            (inputs[0][part], inputs[1][part], *inputs[2:]),
                            dtype="float64",
                        )
                        for part in np.array_split(np.arange(count), min(4, count))
                        if len(part)
                    ]
                )
                row["split_batch_errors"] = error_summary(split, value, atol=1e-12, rtol=1e-10)
            rows.append(row)
    return {
        "reference": "original notebook port",
        "tolerances": {"absolute": 1e-10, "relative": 1e-8},
        "availability": unavailable,
        "results": rows,
    }


def robustness(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Compare edge-case regimes with the stable NumPy formulation and limits."""
    backends, unavailable = _setup(config)
    if "numpy" not in backends:
        raise RuntimeError("robustness requires the NumPy/SciPy backend")
    rng = np.random.default_rng(config["seed"])
    count = config["settings"]["robustness_rows"]
    layers_values = config["settings"]["robustness_layers"]
    rows = []
    # Compare every backend against the stable NumPy formulation across regimes.
    for layers in layers_values:
        for regime in REGIMES:
            inputs = regime_case(rng, count, layers, regime)
            reference = evaluate(backends["numpy"], inputs)
            for name in config["backends"]:
                if name not in backends:
                    rows.append(
                        {
                            "regime": regime,
                            "layers": layers,
                            "backend": name,
                            "status": unavailable[name],
                        }
                    )
                    continue
                value, status = _safe_evaluate(backends[name], inputs, "float64")
                row = {"regime": regime, "layers": layers, "backend": name, "status": status}
                if value is not None:
                    row["errors"] = _errors_with_worst(value, reference, inputs)
                    if name == "original" and regime in {"resonant", "zero_absorption", "thick"}:
                        row["expected_historical_difference"] = True
                rows.append(row)
    analytic = {}
    for name, backend in backends.items():
        value, status = _safe_evaluate(
            backend, (np.array([[0.0]]), np.array([[2.0]]), np.array([]), 1.4, 1.0), "float64"
        )
        analytic[name] = {
            "zero_absorption_value": None if value is None else float(value[0]),
            "expected": 1.0,
            "status": status,
        }
    return {
        "reference": "current NumPy/SciPy stable formulation and analytic limits",
        "availability": unavailable,
        "regimes": list(REGIMES),
        "results": rows,
        "analytic_checks": analytic,
    }


def performance(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Measure solver-only and end-to-end scaling across workloads and dtypes."""
    backends, unavailable = _setup(config)
    rng = np.random.default_rng(config["seed"])
    settings = config["settings"]
    rows = []
    # Separate prepared solver calls from end-to-end input conversion costs.
    for count, layers in settings["performance_workloads"]:
        inputs = ordinary_case(rng, count, layers)
        for dtype in config["dtypes"]:
            for scope in ("solver-only", "end-to-end"):
                for name in config["backends"]:
                    if name not in backends:
                        continue
                    try:
                        prepared = backends[name].prepare(
                            inputs, dtype=dtype, end_to_end=scope == "end-to-end"
                        )
                        timing = measure(
                            prepared.call,
                            warmup=config["warmup"],
                            repetitions=config["repetitions"],
                            synchronize=prepared.synchronize,
                        )
                        rows.append(
                            {
                                "backend": name,
                                "scope": scope,
                                "dtype": dtype,
                                "rows": count,
                                "layers": layers,
                                "throughput_rows_per_second": count / (timing["median"] / 1000),
                                "timing": timing,
                                "status": "ok",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            {
                                "backend": name,
                                "scope": scope,
                                "dtype": dtype,
                                "rows": count,
                                "layers": layers,
                                "status": f"{type(exc).__name__}: {exc}",
                            }
                        )
    for row in rows:
        if "timing" not in row:
            continue
        original = next(
            (
                candidate
                for candidate in rows
                if candidate.get("backend") == "original"
                and candidate.get("scope") == row["scope"]
                and candidate.get("dtype") == row["dtype"]
                and candidate.get("rows") == row["rows"]
                and candidate.get("layers") == row["layers"]
                and "timing" in candidate
            ),
            None,
        )
        row["speedup_vs_original"] = (
            None if original is None else original["timing"]["median"] / row["timing"]["median"]
        )
    return {
        "availability": unavailable,
        "timing_scopes": {
            "solver-only": "prepared inputs; synchronized forward call",
            "end-to-end": "array/tensor conversion, device transfer, forward call, and result transfer",
        },
        "results": rows,
    }


def _batched_calls(
    backend: Backend,
    inputs: Any,
    samples: int,
    wavelengths: int,
    spectra_per_batch: int,
    dtype: str,
    end_to_end: bool = False,
) -> tuple[Callable[[bool], np.ndarray | None], Callable[[], None]]:
    """Prepare batched inference calls and their device synchronizer."""
    # Prepare each chunk once; repeated timed passes then exercise the same batches.
    batches = tuple(spectral_batches(samples, wavelengths, spectra_per_batch))
    prepared = [
        backend.prepare(
            (inputs[0][part], inputs[1][part], *inputs[2:]), dtype=dtype, end_to_end=end_to_end
        )
        for part in batches
    ]

    def run(collect: bool = False) -> np.ndarray | None:
        """Run all prepared chunks and optionally collect their output arrays."""
        values = []
        for item in prepared:
            value = item.call()
            if collect:
                values.append(np.asarray(item.to_numpy(value)))
        for item in prepared:
            item.synchronize()
        return np.concatenate(values) if collect else None

    synchronizer = prepared[0].synchronize if prepared else lambda: None
    return run, synchronizer


def inference(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Compare large-batch and ten-spectrum mini-batch inference paths."""
    backends, unavailable = _setup(config)
    settings = config["settings"]
    samples = settings["inference_samples"]
    wavelengths = settings["inference_wavelengths"]
    layers = 3
    rng = np.random.default_rng(config["seed"])
    mua = rng.uniform(0.001, 0.05, (samples, wavelengths, layers)).reshape(-1, layers)
    musp = rng.uniform(0.5, 3.0, (samples, wavelengths, layers)).reshape(-1, layers)
    inputs = (mua, musp, rng.uniform(0.01, 1.0, layers - 1), 1.4, 1.0)
    # Compare one large pass with repeated mini-batch calls over identical spectra.
    paths = {"large_batch": samples, "mini_batches": settings["inference_mini_batch"]}
    rows = []
    references = {}
    if "numpy" in backends:
        for path, width in paths.items():
            run, _ = _batched_calls(
                backends["numpy"], inputs, samples, wavelengths, width, "float64"
            )
            references[path] = run(collect=True)
    for path, width in paths.items():
        for name in config["backends"]:
            if name not in backends:
                rows.append({"path": path, "backend": name, "status": unavailable[name]})
                continue
            try:
                run, synchronize = _batched_calls(
                    backends[name], inputs, samples, wavelengths, width, config["dtype"]
                )
                value = run(collect=True)
                timing = measure(
                    run,
                    warmup=config["warmup"],
                    repetitions=config["repetitions"],
                    synchronize=synchronize,
                )
                row = {
                    "path": path,
                    "backend": name,
                    "scope": "solver-only",
                    "spectra_per_call": width,
                    "calls_per_pass": (samples + width - 1) // width,
                    "spectra_per_second": samples / (timing["median"] / 1000),
                    "timing": timing,
                    "status": "ok",
                }
                if path in references:
                    row["errors"] = error_summary(value, references[path], atol=1e-10, rtol=1e-8)
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {"path": path, "backend": name, "status": f"{type(exc).__name__}: {exc}"}
                )
    for row in rows:
        if "timing" not in row:
            continue
        original = next(
            (
                candidate
                for candidate in rows
                if candidate.get("backend") == "original"
                and candidate.get("path") == row["path"]
                and "timing" in candidate
            ),
            None,
        )
        row["speedup_vs_original"] = (
            None if original is None else original["timing"]["median"] / row["timing"]["median"]
        )
    return {
        "availability": unavailable,
        "samples": samples,
        "wavelengths_per_spectrum": wavelengths,
        "layers": layers,
        "total_model_evaluations": samples * wavelengths,
        "results": rows,
    }


def high_layer(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Evaluate paper and notebook wavelength grids with 101 layers."""
    backends, unavailable = _setup(config)
    rows = []
    # The two profiles retain their own wavelength grid and deterministic input draw.
    for profile_index, profile_name in enumerate(("paper", "notebook")):
        rng = np.random.default_rng(config["seed"] + profile_index)
        wavelengths, inputs = high_layer_case(rng, profile_name)
        reference = evaluate(backends["numpy"], inputs) if "numpy" in backends else None
        for name in config["backends"]:
            if name not in backends:
                rows.append({"profile": profile_name, "backend": name, "status": unavailable[name]})
                continue
            prepared = backends[name].prepare(inputs, dtype="float64")
            value, status = _safe_evaluate(backends[name], inputs, "float64")
            row = {
                "profile": profile_name,
                "backend": name,
                "wavelengths": len(wavelengths),
                "layers": 101,
                "status": status,
            }
            if value is not None:
                row["checksum"] = float(value.sum())
                row["timing"] = measure(
                    prepared.call,
                    warmup=config["warmup"],
                    repetitions=config["repetitions"],
                    synchronize=prepared.synchronize,
                )
                if reference is not None:
                    row["errors"] = error_summary(value, reference, atol=1e-10, rtol=1e-8)
            rows.append(row)
    return {"availability": unavailable, "results": rows}


def gradients(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Validate directional derivatives and time forward-plus-backward work."""
    backends, unavailable = _setup(config)
    rng = np.random.default_rng(config["seed"])
    rows_count = config["settings"]["gradient_rows"]
    layers = config["settings"]["gradient_layers"]
    inputs = ordinary_case(rng, rows_count, layers)
    mua, musp, thicknesses, nr, outside = inputs
    direction_m = rng.normal(size=mua.shape)
    direction_s = rng.normal(size=musp.shape)
    direction_d = rng.normal(size=thicknesses.shape)
    epsilon = 1e-6
    plus = (
        mua + epsilon * direction_m,
        musp + epsilon * direction_s,
        thicknesses + epsilon * direction_d,
        nr,
        outside,
    )
    minus = (
        mua - epsilon * direction_m,
        musp - epsilon * direction_s,
        thicknesses - epsilon * direction_d,
        nr,
        outside,
    )
    # A centered directional difference provides a common reference for gradient backends.
    reference_directional = float(
        (evaluate(backends["numpy"], plus).sum() - evaluate(backends["numpy"], minus).sum())
        / (2 * epsilon)
    )
    rows = []
    if "cython" in backends:
        from mldt import cython

        start = time.perf_counter_ns()
        result = cython.reflectance_jacobian(
            mua, musp, thicknesses, nr, outside_refractive_index=outside
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        directional = float(
            np.sum(result.d_mua * direction_m)
            + np.sum(result.d_musp * direction_s)
            + np.sum(result.d_thickness * direction_d)
        )
        rows.append(
            {
                "backend": "cython",
                "directional_derivative": directional,
                "reference": reference_directional,
                "absolute_error": abs(directional - reference_directional),
                "forward_backward_ms": elapsed,
                "status": "ok",
            }
        )
    for name in ("torch_cpu", "torch_gpu"):
        backend = backends.get(name)
        if backend is None:
            rows.append({"backend": name, "status": unavailable.get(name, "not requested")})
            continue
        torch = backend.torch
        device = backend.device
        tm = torch.tensor(mua, dtype=torch.float64, device=device, requires_grad=True)
        ts = torch.tensor(musp, dtype=torch.float64, device=device, requires_grad=True)
        td = torch.tensor(thicknesses, dtype=torch.float64, device=device, requires_grad=True)

        def forward_backward(
            tm: Any = tm, ts: Any = ts, td: Any = td, backend: Backend = backend
        ) -> None:
            """Clear old gradients and run one forward/backward pass."""
            for tensor in (tm, ts, td):
                tensor.grad = None
            backend.function(tm, ts, td, nr, outside_refractive_index=outside).sum().backward()

        forward_backward()
        backend._synchronizer()()
        directional = float(
            (tm.grad * torch.as_tensor(direction_m, device=device)).sum()
            + (ts.grad * torch.as_tensor(direction_s, device=device)).sum()
            + (td.grad * torch.as_tensor(direction_d, device=device)).sum()
        )
        timing = measure(
            forward_backward,
            warmup=config["warmup"],
            repetitions=config["repetitions"],
            synchronize=backend._synchronizer(),
        )
        rows.append(
            {
                "backend": name,
                "directional_derivative": directional,
                "reference": reference_directional,
                "absolute_error": abs(directional - reference_directional),
                "timing": timing,
                "status": "ok",
            }
        )
    return {"availability": unavailable, "finite_difference_epsilon": epsilon, "results": rows}


def _external_optical_properties(
    payload: dict[str, Any], sample_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Convert one fixture sample into wavelength-dependent solver inputs."""
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=float)
    layers = np.asarray(payload["parameters"][sample_index][3:], dtype=float).reshape(3, 9)
    library = payload["optical_library"]
    lw = np.asarray(library["wavelength_nm"], dtype=float)
    absorption = library["absorption_mm_inverse"]
    components = [
        np.interp(wavelengths, lw, absorption[key])
        for key in ("oxyhemoglobin", "deoxyhemoglobin", "melanin", "water")
    ]
    hbo2, hb, melanin, water = components
    mua = np.empty((len(wavelengths), 3))
    musp = np.empty_like(mua)
    for index, (_d, blood, saturation, mel, wat, a, b, _g, _n) in enumerate(layers):
        mua[:, index] = (
            mel * melanin + saturation * blood * hbo2 + (1 - saturation) * blood * hb + wat * water
        )
        musp[:, index] = a * (wavelengths / 500) ** -b
    return mua, musp, layers[:2, 0], float(layers[0, 8]), 1.0


def _mcmlnet_prediction(payload: dict[str, Any], model_config: dict[str, Any]) -> Any:
    """Load the configured model and predict reflectance for the reference fixture."""
    if not all(model_config.values()):
        return None, None, "not run: model configuration not provided"
    try:
        import torch
        from mcmlnet.utils.convenience import (
            batch_inputs_to_three_layer_model,
            prepare_surrogate_model_data,
        )

        from .mcmlnet import load_model

        model, preprocessor, config = load_model(
            model_config["model_dir"],
            model_config["model_version"],
            model_config["checkpoint_path"],
        )
        predictions, uncertainties = [], []
        wavelengths = np.asarray(payload["wavelength_nm"], dtype=float)
        library = payload["optical_library"]
        lw = np.asarray(library["wavelength_nm"], dtype=float)
        absorption = library["absorption_mm_inverse"]
        spectra = {
            key: np.interp(wavelengths, lw, absorption[key])
            for key in ("oxyhemoglobin", "deoxyhemoglobin", "melanin", "water")
        }
        for sample_index in range(len(payload["parameters"])):
            layers = np.asarray(payload["parameters"][sample_index][3:], dtype=float).reshape(3, 9)
            mua = np.empty((len(wavelengths), 3))
            musp = np.empty_like(mua)
            for index, (_d, blood, saturation, mel, wat, a, b, _g, _n) in enumerate(layers):
                mua[:, index] = (
                    mel * spectra["melanin"]
                    + saturation * blood * spectra["oxyhemoglobin"]
                    + (1 - saturation) * blood * spectra["deoxyhemoglobin"]
                    + wat * spectra["water"]
                )
                musp[:, index] = a * (wavelengths / 500) ** -b
            g = np.broadcast_to(layers[:, 7], mua.shape).copy()
            n = np.broadcast_to(layers[:, 8], mua.shape).copy()
            thickness = np.broadcast_to(layers[:, 0] / 1000, mua.shape).copy()
            data = batch_inputs_to_three_layer_model(
                mua * 1000, musp * 1000 / (1 - g), g, n, thickness
            )
            data = torch.from_numpy(data).unsqueeze(1).float()
            data = torch.cat((data, torch.zeros_like(data)[..., [0]]), dim=-1)
            data = prepare_surrogate_model_data(preprocessor, config, data)[..., :-1]
            with torch.inference_mode():
                if hasattr(model, "predict_outputs"):
                    prediction = model.predict_outputs(data, return_std=True)
                    mean, deviation = prediction.mean, prediction.std
                elif hasattr(model, "predict_mean_std"):
                    mean, deviation = model.predict_mean_std(data)
                else:
                    raise TypeError("mcmlnet model does not expose uncertainty outputs")
            predictions.append(mean.detach().cpu().numpy().reshape(-1))
            uncertainties.append(deviation.detach().cpu().numpy().reshape(-1))
        return np.asarray(predictions), np.asarray(uncertainties), "ok"
    except Exception as exc:  # noqa: BLE001 - optional publication dependency
        return None, None, f"not run: {type(exc).__name__}: {exc}"


def external(config: dict[str, Any], _output_dir: Path) -> dict[str, Any]:
    """Compare solver and optional learned predictions with external references."""
    """Compare solver spectra with MCX, optional MCML, and optional mcmlnet."""
    backends, unavailable = _setup(config)
    root = Path(__file__).resolve().parents[3]
    reference_path = root / "data" / "mcx_reference.json"
    # Load the pinned reference fixture before comparing analytic and learned predictions.
    if not reference_path.is_file():
        raise FileNotFoundError(f"missing external reference: {reference_path}")
    payload = json.loads(reference_path.read_text())
    mcx = np.asarray(payload["reflectance_mcx"], dtype=float)
    surface_index = payload["parameter_names"].index("n1")
    surface_refractive_indices = np.asarray(payload["parameters"], dtype=float)[:, surface_index]
    outside_refractive_index = 1.0
    specular_reflectance = (
        (surface_refractive_indices - outside_refractive_index)
        / (surface_refractive_indices + outside_refractive_index)
    ) ** 2
    predictions = {name: np.empty_like(mcx) for name in backends}
    for index in range(len(mcx)):
        inputs = _external_optical_properties(payload, index)
        for name, backend in backends.items():
            predictions[name][index] = evaluate(backend, inputs)
    rows = [
        {
            "backend": name,
            "versus_mcx": error_summary(value, mcx, atol=0.0, rtol=0.0),
            "prediction": value.tolist(),
            "status": "ok",
        }
        for name, value in predictions.items()
    ]
    simulated_path = root / "data" / "mcml_reference_1m.json"
    mcml = json.loads(simulated_path.read_text()) if simulated_path.is_file() else None
    surrogate, uncertainty, surrogate_status = _mcmlnet_prediction(
        payload, config.get("external_model", {})
    )
    return {
        "availability": unavailable,
        "reference": str(reference_path),
        "wavelength_nm": payload["wavelength_nm"],
        "mcx": mcx.tolist(),
        "mcml": None if mcml is None else mcml.get("reflectance_mcml"),
        "specular_reflectance": specular_reflectance.tolist(),
        "results": rows,
        "mcmlnet": {
            "status": surrogate_status,
            "prediction": None if surrogate is None else surrogate.tolist(),
            "standard_deviation": None if uncertainty is None else uncertainty.tolist(),
        },
    }


def profile(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Capture and save profiler statistics for each available backend."""
    """Create cProfile artifacts for requested available backends."""
    backends, unavailable = _setup(config)
    inputs = ordinary_case(np.random.default_rng(config["seed"]), 96, 20)
    rows = []
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    # Save profiler output per backend so expensive traces remain inspectable.
    for name, backend in backends.items():
        prepared = backend.prepare(inputs)
        artifact = profile_dir / f"{name}.prof"
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        for _ in range(config["repetitions"]):
            prepared.call()
        prepared.synchronize()
        profiler.disable()
        elapsed = time.perf_counter() - start
        profiler.dump_stats(artifact)
        report = profile_dir / f"{name}.txt"
        with report.open("w") as stream:
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
            stats.print_stats(60)
        rows.append(
            {
                "backend": name,
                "elapsed_seconds": elapsed,
                "calls": config["repetitions"],
                "profile": str(artifact),
                "report": str(report),
                "status": "ok",
            }
        )
    return {
        "availability": unavailable,
        "gpu_note": "cProfile reports CPU launch/dispatch time; use a device profiler for GPU kernels.",
        "results": rows,
    }


STAGES = {
    "equivalence": equivalence,
    "robustness": robustness,
    "performance": performance,
    "inference": inference,
    "high-layer": high_layer,
    "gradients": gradients,
    "external": external,
    "profile": profile,
}
