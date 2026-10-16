#!/usr/bin/env python3
"""Create a Markdown report and focused figures from unified suite results."""

from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path

import numpy as np

from scripts.evaluations.benchmark_core.schema import read_json

LABELS = {
    "original": "Original notebook",
    "numpy": "NumPy/SciPy",
    "cython": "Cython",
    "torch_cpu": "PyTorch CPU",
    "torch_gpu": "PyTorch GPU",
}
COLORS = {
    "original": "#D55E00",
    "numpy": "#CC79A7",
    "cython": "#0072B2",
    "torch_cpu": "#56B4E9",
    "torch_gpu": "#6A3D9A",
}

MCML_COLOR = "#E69F00"
MCMLNET_COLOR = "#009E73"


def _plt() -> Any:
    """Import and return Matplotlib's plotting interface."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("analysis requires matplotlib; install the dev extra") from exc
    return plt


def _successful(stage: dict[str, Any] | None) -> bool:
    """Return whether a stage completed successfully."""
    return stage and stage.get("status") == "complete"


def accuracy_plot(
    stage: dict[str, Any], output: Path, title: str, *, regime_groups: bool = False
) -> None:
    """Plot maximum errors and tolerance failures for an accuracy stage."""
    plt = _plt()
    # Plot rows with numeric comparisons; failed evaluations are marked separately.
    rows = [row for row in stage["results"] if "errors" in row]
    groups = list(
        dict.fromkeys((row.get("regime") or row.get("workload"), row.get("layers")) for row in rows)
    )
    if regime_groups:
        regimes = list(stage.get("regimes", dict.fromkeys(group[0] for group in groups)))
        layers = list(stage.get("config", {}).get("settings", {}).get("robustness_layers", ()))
        if not layers:
            layers = list(dict.fromkeys(group[1] for group in groups))
        groups = [
            (regime, layer) for regime in regimes for layer in layers if (regime, layer) in groups
        ]
    names = list(dict.fromkeys(row["backend"] for row in rows))
    fig, (maximum_axis, failure_axis) = plt.subplots(
        1, 2, figsize=(18 if regime_groups else 14, 6.2 if regime_groups else 5.5)
    )
    x = np.arange(len(groups))
    for name in names:
        selected = {
            (row.get("regime") or row.get("workload"), row.get("layers")): row
            for row in rows
            if row["backend"] == name
        }
        maximum_axis.plot(
            x,
            [
                max(selected[group]["errors"]["max_abs_error"] or 0, 1e-18)
                if group in selected
                else np.nan
                for group in groups
            ],
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
        failure_axis.plot(
            x,
            [
                selected[group]["errors"]["tolerance_failures"] if group in selected else np.nan
                for group in groups
            ],
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
    maximum_axis.set_yscale("log")
    maximum_axis.set_ylabel("Max |backend − NumPy|" if regime_groups else "Maximum absolute error")
    failure_count = stage.get("config", {}).get("settings", {}).get("robustness_rows")
    failure_axis.set_ylabel(
        f"Tolerance failures (of {failure_count:,})" if failure_count else "Tolerance failures"
    )
    for axis in (maximum_axis, failure_axis):
        if regime_groups:
            axis.set_xticks(x, [str(group[1]) for group in groups], fontsize=8)
            regime_names = list(dict.fromkeys(group[0] for group in groups))
            per_regime = len(layers)
            regime_labels = {
                "low_absorption": "low μa",
                "zero_absorption": "zero μa",
            }
            for index, regime in enumerate(regime_names):
                center = index * per_regime + (per_regime - 1) / 2
                axis.text(
                    center,
                    -0.16,
                    regime_labels.get(regime, regime.replace("_", " ")),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                )
                if index:
                    axis.axvline(index * per_regime - 0.5, color="0.55", lw=0.8, alpha=0.7)
        else:
            labels = [f"{group[0]}\nL={group[1]}" for group in groups]
            axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(alpha=0.25)
    if regime_groups:
        for name in names:
            unavailable_status = stage.get("availability", {}).get(name)
            failed = {
                (row.get("regime"), row.get("layers"))
                for row in stage["results"]
                if row["backend"] == name
                and "errors" not in row
                and row.get("status") not in {None, "ok", unavailable_status}
            }
            failed_x = [index for index, group in enumerate(groups) if group in failed]
            if failed_x:
                for axis in (maximum_axis, failure_axis):
                    axis.scatter(
                        failed_x,
                        [0.97] * len(failed_x),
                        transform=axis.get_xaxis_transform(),
                        marker="x",
                        s=70,
                        linewidths=2,
                        color=COLORS[name],
                        zorder=6,
                        clip_on=False,
                    )
    maximum_axis.legend(fontsize=8)
    fig.suptitle(title)
    if regime_groups:
        fig.subplots_adjust(bottom=0.25, top=0.86, wspace=0.22)
        fig.text(0.5, 0.055, "Layer count within each regime", ha="center")
        fig.text(
            0.5,
            0.91,
            "× marks an evaluation exception; no numeric error or failure count was produced",
            ha="center",
            fontsize=9,
        )
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def timing_plot(stage: dict[str, Any], output: Path, title: str, group_key: str = "rows") -> None:
    """Plot float64 solver-only median timings by workload group."""
    plt = _plt()
    # Keep solver-only float64 timings comparable across backends.
    rows = [row for row in stage["results"] if "timing" in row and row.get("status") == "ok"]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    names = list(dict.fromkeys(row["backend"] for row in rows))
    for name in names:
        selected = [
            row
            for row in rows
            if row["backend"] == name
            and row.get("scope", "solver-only") == "solver-only"
            and row.get("dtype", "float64") == "float64"
        ]
        labels = [str(row.get(group_key, row.get("profile", row.get("path")))) for row in selected]
        axis.plot(
            labels,
            [row["timing"]["median"] for row in selected],
            marker="o",
            label=LABELS[name],
            color=COLORS[name],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Median runtime (ms, log scale)")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def inference_plot(stage: dict[str, Any], output: Path) -> None:
    """Plot latency and throughput for representative inference paths."""
    """Plot latency and throughput for both representative inference paths."""
    plt = _plt()
    rows = [row for row in stage["results"] if "timing" in row]
    names = list(stage.get("config", {}).get("backends", []))
    paths = ("large_batch", "mini_batches")
    x = np.arange(len(names))
    width = 0.36
    fig, (latency, throughput) = plt.subplots(1, 2, figsize=(13, 5.5))
    for index, path in enumerate(paths):
        selected = {(row["backend"], row["path"]): row for row in rows}
        offset = (index - 0.5) * width
        latency.bar(
            x + offset,
            [
                selected[(name, path)]["timing"]["median"] if (name, path) in selected else np.nan
                for name in names
            ],
            width,
            label=path.replace("_", " "),
        )
        throughput.bar(
            x + offset,
            [
                selected[(name, path)]["spectra_per_second"] if (name, path) in selected else np.nan
                for name in names
            ],
            width,
        )
    for axis in (latency, throughput):
        axis.set_yscale("log")
        axis.set_xticks(x, [LABELS[name] for name in names], rotation=25, ha="right")
        axis.grid(axis="y", which="both", alpha=0.25)
    latency.set_ylabel("Full-pass latency (ms)")
    throughput.set_ylabel("Spectra per second")
    latency.legend()
    fig.suptitle("Representative three-layer inference workload")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def gradient_plot(stage: dict[str, Any], output: Path) -> None:
    """Plot directional derivative agreement and gradient runtime."""
    """Plot directional-derivative error and forward-plus-backward timing."""
    plt = _plt()
    rows = [row for row in stage["results"] if row.get("status") == "ok"]
    fig, (error_axis, runtime_axis) = plt.subplots(1, 2, figsize=(11, 4.8))
    names = [row["backend"] for row in rows]
    error_axis.bar(
        [LABELS[name] for name in names],
        [max(row["absolute_error"], 1e-18) for row in rows],
        color=[COLORS[name] for name in names],
    )
    runtime_axis.bar(
        [LABELS[name] for name in names],
        [
            row.get("forward_backward_ms", row.get("timing", {}).get("median", np.nan))
            for row in rows
        ],
        color=[COLORS[name] for name in names],
    )
    error_axis.set_yscale("log")
    runtime_axis.set_yscale("log")
    error_axis.set_ylabel("Directional derivative absolute error")
    runtime_axis.set_ylabel("Forward + backward runtime (ms)")
    for axis in (error_axis, runtime_axis):
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def external_plot(
    stage: dict[str, Any],
    output: Path,
    *,
    difference: bool = False,
    add_specular: bool = False,
) -> None:
    """Plot external reference and model predictions by wavelength."""
    """Plot all nine external spectra, optionally as residuals or with specular terms."""
    plt = _plt()
    wavelength = np.asarray(stage["wavelength_nm"], dtype=float)
    mcx = np.asarray(stage["mcx"], dtype=float)
    sample_count = len(mcx)
    if sample_count != 9:
        raise ValueError(f"external 3x3 plots require nine samples, received {sample_count}")

    # Older suite documents predate this field. The publication fixture uses
    # tissue n=1.4 and outside n=1.0 for every sample.
    specular = np.asarray(
        stage.get("specular_reflectance", [((1.4 - 1.0) / (1.4 + 1.0)) ** 2] * sample_count),
        dtype=float,
    )[:, None]
    adjustment = specular if add_specular else 0.0
    groups = ("homogeneous",) * 3 + ("two-layer",) * 3 + ("three-layer",) * 3
    letters = "abcdefghi"
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)

    def plotted(values: np.ndarray) -> np.ndarray:
        """Apply the plot's selected absolute or difference transformation."""
        return values - mcx if difference else values

    handles = []
    labels = []
    for index, axis in enumerate(axes.flat):
        if difference:
            axis.axhline(0, color="#555555", linewidth=1)
        else:
            line = axis.plot(wavelength, mcx[index], color="black", linewidth=2, label="MCX (10k)")[
                0
            ]
            if index == 0:
                handles.append(line)
                labels.append(line.get_label())

        if stage.get("mcml") is not None:
            mcml = np.asarray(stage["mcml"], dtype=float) + adjustment
            label = "MCML (1M)" + (" + specular" if add_specular else "")
            if difference:
                label += " − MCX (10k)"
            line = axis.plot(wavelength, plotted(mcml)[index], color=MCML_COLOR, label=label)[0]
            if index == 0:
                handles.append(line)
                labels.append(line.get_label())

        for row in stage["results"]:
            prediction = np.asarray(row["prediction"], dtype=float)
            line = axis.plot(
                wavelength,
                plotted(prediction)[index],
                label=LABELS[row["backend"]],
                color=COLORS[row["backend"]],
                linestyle="--" if row["backend"] == "original" else "-.",
            )[0]
            if index == 0:
                handles.append(line)
                labels.append(line.get_label())

        surrogate_payload = stage.get("mcmlnet", {})
        if surrogate_payload.get("prediction") is not None:
            surrogate = np.asarray(surrogate_payload["prediction"], dtype=float) + adjustment
            deviation = np.asarray(surrogate_payload["standard_deviation"], dtype=float)
            label = "mcmlnet mean" + (" + specular" if add_specular else "")
            line = axis.plot(
                wavelength,
                plotted(surrogate)[index],
                color=MCMLNET_COLOR,
                linestyle=":",
                label=label,
            )[0]
            band = axis.fill_between(
                wavelength,
                plotted(surrogate - deviation)[index],
                plotted(surrogate + deviation)[index],
                color=MCMLNET_COLOR,
                alpha=0.25,
                linewidth=0,
                label="mcmlnet ±1σ",
            )
            if index == 0:
                handles.extend((line, band))
                labels.extend((line.get_label(), band.get_label()))

        axis.set_title(f"({letters[index]}) {groups[index]}", loc="left")
        axis.grid(alpha=0.25)
        if index >= 6:
            axis.set_xlabel("Wavelength (nm)")
        if index % 3 == 0:
            axis.set_ylabel(r"$R_d - R_{d,\mathrm{MCX}}$" if difference else r"$R_d$")

    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze(payload: dict[str, Any], output_dir: Path) -> Path:
    """Generate the Markdown report and applicable static figures."""
    if payload.get("kind") != "mldt-evaluation-suite":
        raise ValueError("input is not a unified MLDT suite result")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Plot specifications map successful stages to report figure files.
    figures = []
    stages = payload["stages"]
    if _successful(stages.get("external")):
        (output_dir / "external.png").unlink(missing_ok=True)
    plot_specs = (
        (
            "equivalence",
            "equivalence.png",
            lambda s, p: accuracy_plot(s, p, "Agreement with original notebook on regular inputs"),
        ),
        (
            "robustness",
            "robustness.png",
            lambda s, p: accuracy_plot(
                s, p, "Stable-formulation robustness by regime", regime_groups=True
            ),
        ),
        (
            "performance",
            "performance.png",
            lambda s, p: timing_plot(s, p, "Solver-only runtime scaling"),
        ),
        ("inference", "inference.png", inference_plot),
        (
            "high-layer",
            "high-layer.png",
            lambda s, p: timing_plot(s, p, "101-layer workloads", "profile"),
        ),
        ("gradients", "gradients.png", gradient_plot),
        (
            "external",
            "external-absolute.png",
            lambda s, p: external_plot(s, p),
        ),
        (
            "external",
            "external-difference.png",
            lambda s, p: external_plot(s, p, difference=True),
        ),
        (
            "external",
            "external-absolute-specular.png",
            lambda s, p: external_plot(s, p, add_specular=True),
        ),
        (
            "external",
            "external-difference-specular.png",
            lambda s, p: external_plot(s, p, difference=True, add_specular=True),
        ),
    )
    for stage_name, filename, plotter in plot_specs:
        stage = stages.get(stage_name)
        if _successful(stage):
            path = output_dir / filename
            plotter(stage, path)
            figures.append((stage_name, filename))
    # Build the report from the same payload, including availability and provenance.
    report = [
        "# MLDT unified evaluation report",
        "",
        f"Profile: `{payload['config']['profile']}`  ",
        f"Schema version: `{payload['schema_version']}`  ",
        f"Created: `{payload['created_at']}`",
        "",
        "## Backend availability",
        "",
    ]
    requested = payload["config"]["backends"]
    unavailable = {}
    for stage in stages.values():
        unavailable.update(stage.get("availability", {}))
    report.extend([f"- {LABELS[name]}: {unavailable.get(name, 'available')}" for name in requested])
    report.extend(
        [
            "",
            "## Method and interpretation",
            "",
            "Maintained backends reproduce the original on its intended regular domain while deliberately correcting its singular and limiting behavior. The original notebook port is a historical equivalence baseline, not the numerical oracle for resonance, zero absorption, or optically thick limits.",
            "",
            "Solver-only timings use prepared inputs and synchronized forward calls. End-to-end timings additionally include conversion, device transfer, and result transfer.",
            "",
            "## Stage status",
            "",
        ]
    )
    report.extend(
        [
            f"- `{name}`: {stage.get('status', 'unknown')}{' — ' + stage['error'] if stage.get('error') else ''}"
            for name, stage in stages.items()
        ]
    )
    if figures:
        report.extend(["", "## Figures", ""])
        report.extend(
            [
                f"### {name.replace('-', ' ').title()}\n\n![{name}]({filename})\n"
                for name, filename in figures
            ]
        )
    report.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```json",
            __import__("json").dumps(payload["environment"], indent=2),
            "```",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report))
    return output_dir / "report.md"


def main() -> None:
    """Analyze one consolidated suite result from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent
    print(analyze(read_json(args.input), output))


if __name__ == "__main__":
    main()
