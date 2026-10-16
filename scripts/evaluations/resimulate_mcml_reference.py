#!/usr/bin/env python3
"""Re-simulate the qualitative MCX fixture with the local GPU MCML command."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = REPOSITORY_ROOT / "data" / "mcx_reference.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "mcml_reference_1m.json"
EXPECTED_COLUMNS = [
    "ID",
    "Specular",
    "Diffuse",
    "Absorbed",
    "Transmittance",
    "Penetration",
]


def optical_properties(payload: dict) -> tuple[np.ndarray, ...]:
    """Convert the stored layer parameters and library into per-wavelength MCML inputs."""
    # Interpolate constituent spectra, then derive each layer's optical inputs.
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=float)
    library = payload["optical_library"]
    library_wavelengths = np.asarray(library["wavelength_nm"], dtype=float)
    absorption = library["absorption_mm_inverse"]
    hbo2 = np.interp(wavelengths, library_wavelengths, absorption["oxyhemoglobin"])
    hb = np.interp(wavelengths, library_wavelengths, absorption["deoxyhemoglobin"])
    melanin = np.interp(wavelengths, library_wavelengths, absorption["melanin"])
    water = np.interp(wavelengths, library_wavelengths, absorption["water"])

    all_mua, all_mus, all_g, all_n, all_thickness_cm = [], [], [], [], []
    for row in payload["parameters"]:
        layers = np.asarray(row[3:], dtype=float).reshape(3, 9)
        mua_mm = np.empty((wavelengths.size, 3), dtype=float)
        musp_mm = np.empty_like(mua_mm)
        for index, (_thickness, blood, saturation, mel, wat, a, b, _g, _n) in enumerate(layers):
            mua_mm[:, index] = (
                mel * melanin
                + saturation * blood * hbo2
                + (1.0 - saturation) * blood * hb
                + wat * water
            )
            musp_mm[:, index] = a * (wavelengths / 500.0) ** (-b)

        g = np.broadcast_to(layers[:, 7], mua_mm.shape).copy()
        n = np.broadcast_to(layers[:, 8], mua_mm.shape).copy()
        # The supplied third layer represents a semi-infinite substrate. Use 20 cm,
        # as requested, while retaining the first two finite layer thicknesses.
        thickness_cm = layers[:, 0] / 10.0
        thickness_cm[-1] = 20.0
        all_mua.append(mua_mm * 10.0)  # MCML absorption is cm^-1.
        all_mus.append((musp_mm / (1.0 - g)) * 10.0)  # MCML scattering is cm^-1.
        all_g.append(g)
        all_n.append(n)
        all_thickness_cm.append(thickness_cm)

    return tuple(
        np.asarray(values) for values in (all_mua, all_mus, all_g, all_n, all_thickness_cm)
    )


def write_mci(
    path: Path,
    output_stem: str,
    photons: int,
    mua: np.ndarray,
    mus: np.ndarray,
    g: np.ndarray,
    refractive_index: np.ndarray,
    thickness_cm: np.ndarray,
) -> None:
    """Write a multi-run MCML input file for all fixture wavelengths."""
    wavelengths, layers = mua.shape
    lines = ["1.0 # file version", f"{wavelengths} # number of runs"]
    # MCML accepts one material stack per run in this multi-run input file.
    for wavelength_index in range(wavelengths):
        lines.extend(
            [
                f"{output_stem}_{wavelength_index:03d}.mco A # output filename, ASCII",
                f"{photons} # No. of photons",
                "0.002 2.0 # dz, dr",
                "500 1 1 # No. of dz, dr & da",
                f"{layers} # No. of layers",
                "1.0 # n for medium above",
            ]
        )
        for layer_index in range(layers):
            lines.append(
                f"{refractive_index[wavelength_index, layer_index]:.10g} "
                f"{mua[wavelength_index, layer_index]:.10g} "
                f"{mus[wavelength_index, layer_index]:.10g} "
                f"{g[wavelength_index, layer_index]:.10g} "
                f"{thickness_cm[layer_index]:.10g}"
            )
        lines.append("1.0 # n for medium below")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_reflectance(path: Path, expected_count: int) -> np.ndarray:
    """Read diffuse reflectance values from one MCML ASCII output."""
    # Validate output shape and physical range before accepting simulation values.
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected MCML output header in {path}: {reader.fieldnames!r}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} MCML output rows in {path}, got {len(rows)}")
    values = np.asarray([float(row["Diffuse"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"MCML produced invalid diffuse reflectance values in {path}")
    return values


def main() -> None:
    """Run the local MCML executable and persist a provenance-tagged fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mcml", default="MCML", help="local MCML executable (default: MCML on PATH)"
    )
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    if args.photons <= 0 or args.gpus <= 0 or args.seed < 0:
        parser.error("photons and gpus must be positive; seed must be non-negative")
    executable = shutil.which(args.mcml)
    if executable is None:
        parser.error(f"could not find MCML executable: {args.mcml}")

    # Prepare the fixture and convert its parameters to MCML's units.
    payload = json.loads(args.reference.read_text(encoding="utf-8"))
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=float)
    mua, mus, g, refractive_index, thickness_cm = optical_properties(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reflectance = np.empty((len(payload["parameters"]), wavelengths.size), dtype=float)
    case_runtime_seconds = []

    with tempfile.TemporaryDirectory(prefix="mcml-reference-") as scratch_name:
        # Keep transient MCML inputs and outputs out of the fixture directory.
        scratch = Path(scratch_name)
        for sample_index in range(len(payload["parameters"])):
            input_path = scratch / f"sample-{sample_index:02d}.mci"
            output_path = scratch / f"sample-{sample_index:02d}.csv"
            write_mci(
                input_path,
                f"sample-{sample_index:02d}",
                args.photons,
                mua[sample_index],
                mus[sample_index],
                g[sample_index],
                refractive_index[sample_index],
                thickness_cm[sample_index],
            )
            command = [
                executable,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--seed",
                str(args.seed + sample_index),
                "--n_gpus",
                str(args.gpus),
            ]
            print(
                f"Simulating case {sample_index + 1}/{len(reflectance)} "
                f"({wavelengths.size} wavelengths)",
                flush=True,
            )
            try:
                start = time.perf_counter()
                subprocess.run(command, check=True)
                case_runtime_seconds.append(time.perf_counter() - start)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"MCML failed for case {sample_index} ({input_path}); "
                    "check that its CUDA device is available"
                ) from exc
            reflectance[sample_index] = read_reflectance(output_path, wavelengths.size)

    # Preserve source provenance and record how this replacement reference was produced.
    result = dict(payload)
    result["provenance"] = dict(payload.get("provenance", {}))
    upstream_source = result["provenance"].get("publication_source")
    upstream_source_sha256 = result["provenance"].get("publication_source_sha256")
    if not upstream_source or not upstream_source_sha256:
        upstream_source = result["provenance"].get("source")
        upstream_source_sha256 = result["provenance"].get("source_sha256")
    source_layers = np.asarray(payload["parameters"][0][3:], dtype=float).reshape(3, 9)
    result["provenance"].update(
        {
            "source": args.reference.name,
            "source_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "mcml_photon_count": args.photons,
            "mcml_executable": args.mcml,
            "mcml_seed_start": args.seed,
            "mcml_gpus": args.gpus,
            "mcml_layer_thickness_cm": [
                *(source_layers[:2, 0] / 10.0).tolist(),
                20.0,
            ],
            "notes": (
                "Re-simulated from the stored optical-property library with local MCML; "
                "diffuse reflectance is the MCML Diffuse output."
            ),
        }
    )
    if upstream_source and upstream_source_sha256:
        result["provenance"].update(
            {
                "publication_source": upstream_source,
                "publication_source_sha256": upstream_source_sha256,
            }
        )
    result.pop("reflectance_mcx", None)
    result["reflectance_mcml"] = reflectance.tolist()
    result["simulation"] = {
        "engine": "MCML",
        "photons_per_wavelength": args.photons,
        "semi_infinite_layer_thickness_cm": 20.0,
        "seed_start": args.seed,
        "gpus": args.gpus,
        "mcml_runtime_seconds": sum(case_runtime_seconds),
        "mcml_case_runtime_seconds": case_runtime_seconds,
    }
    # Replace the destination only after the complete fixture has been serialized.
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
