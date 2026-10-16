"""Shared implementation for the unified MLDT evaluation suite."""

# Shared identifiers keep CLI validation, workers, and reports in sync.
SCHEMA_VERSION = 1
BACKEND_NAMES = ("original", "numpy", "cython", "torch_cpu", "torch_gpu")
STAGE_NAMES = (
    "equivalence",
    "robustness",
    "performance",
    "inference",
    "high-layer",
    "gradients",
    "external",
    "profile",
)
