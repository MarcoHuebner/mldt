"""Discovery and normalized adapters for all primary MLDT backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import BACKEND_NAMES


@dataclass
class PreparedCall:
    """Prepared backend invocation plus synchronization and result conversion."""

    call: Callable[[], Any]
    synchronize: Callable[[], None] = lambda: None
    to_numpy: Callable[[Any], np.ndarray] = np.asarray


@dataclass
class Backend:
    """Normalized interface to one solver implementation and device."""

    name: str
    device: str
    function: Callable[..., Any]
    torch: Any = None

    def prepare(
        self, inputs: Any, *, dtype: str = "float64", end_to_end: bool = False
    ) -> PreparedCall:
        """Prepare backend inputs and return a callable for the selected timing scope."""
        if dtype == "float32" and self.torch is None:
            raise ValueError(f"{self.name} does not provide native float32 execution")
        mua, musp, thicknesses, refractive_index, outside = inputs
        np_dtype = np.float32 if dtype == "float32" else np.float64
        if self.torch is None:
            if end_to_end:
                # Include host array conversion in this timing scope.

                def call() -> Any:
                    """Convert host inputs and execute the backend."""
                    m, s, d = (
                        np.array(value, dtype=np_dtype, copy=True)
                        for value in (mua, musp, thicknesses)
                    )
                    return self.function(
                        m, s, d, refractive_index, outside_refractive_index=outside
                    )

                return PreparedCall(call=call)

            # Convert once so solver-only timing measures the call, not preparation.
            arrays = tuple(np.asarray(value, dtype=np_dtype) for value in (mua, musp, thicknesses))

            def call() -> Any:
                """Execute the backend using prepared NumPy arrays."""
                m, s, d = arrays
                return self.function(m, s, d, refractive_index, outside_refractive_index=outside)

            return PreparedCall(call=call)

        torch = self.torch
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        device = self.device
        if end_to_end:
            # Include tensor creation, transfers, execution, and result transfer.

            def call() -> np.ndarray:
                """Transfer inputs, execute, and return the host result."""
                tm = torch.as_tensor(mua, dtype=torch_dtype, device=device)
                ts = torch.as_tensor(musp, dtype=torch_dtype, device=device)
                td = torch.as_tensor(thicknesses, dtype=torch_dtype, device=device)
                with torch.inference_mode():
                    result = self.function(
                        tm, ts, td, refractive_index, outside_refractive_index=outside
                    )
                return result.cpu().numpy()

            return PreparedCall(call=call, synchronize=self._synchronizer())

        # Prepare tensors outside the timed call for solver-only measurements.
        tm = torch.as_tensor(mua, dtype=torch_dtype, device=device)
        ts = torch.as_tensor(musp, dtype=torch_dtype, device=device)
        td = torch.as_tensor(thicknesses, dtype=torch_dtype, device=device)

        def call() -> Any:
            """Execute using tensors prepared on the target device."""
            with torch.inference_mode():
                return self.function(tm, ts, td, refractive_index, outside_refractive_index=outside)

        return PreparedCall(
            call=call,
            synchronize=self._synchronizer(),
            to_numpy=lambda value: value.detach().cpu().numpy(),
        )

    def _synchronizer(self) -> Callable[[], None]:
        """Return the device synchronization function when required."""
        if self.device == "cuda":
            return self.torch.cuda.synchronize
        return lambda: None


def discover(requested: list[str]) -> tuple[dict[str, Backend], dict[str, str]]:
    """Discover requested backends without failing on optional dependencies."""
    unknown = sorted(set(requested) - set(BACKEND_NAMES))
    if unknown:
        raise ValueError("unknown backend(s): " + ", ".join(unknown))
    available: dict[str, Backend] = {}
    unavailable: dict[str, str] = {}
    # Optional backends are reported as unavailable instead of aborting discovery.
    if "original" in requested:
        try:
            from mldt.original import reflectance

            available["original"] = Backend("original", "cpu", reflectance)
        except ImportError as exc:
            unavailable["original"] = str(exc)
    if "numpy" in requested:
        from mldt.numpy import reflectance

        available["numpy"] = Backend("numpy", "cpu", reflectance)
    if "cython" in requested:
        from mldt import cython

        if cython.is_available():
            available["cython"] = Backend("cython", "cpu", cython.reflectance)
        else:
            unavailable["cython"] = "native extension is not built"
    torch_names = set(requested) & {"torch_cpu", "torch_gpu"}
    if torch_names:
        try:
            import torch

            from mldt.torch import reflectance
        except ImportError as exc:
            for name in torch_names:
                unavailable[name] = str(exc)
        else:
            if "torch_cpu" in torch_names:
                available["torch_cpu"] = Backend("torch_cpu", "cpu", reflectance, torch)
            if "torch_gpu" in torch_names:
                if torch.cuda.is_available():
                    available["torch_gpu"] = Backend("torch_gpu", "cuda", reflectance, torch)
                else:
                    unavailable["torch_gpu"] = "CUDA is unavailable"
    return available, unavailable


def evaluate(backend: Backend, inputs: Any, *, dtype: str = "float64") -> np.ndarray:
    """Evaluate prepared inputs and normalize the result to float64 NumPy."""
    prepared = backend.prepare(inputs, dtype=dtype)
    value = prepared.call()
    prepared.synchronize()
    return np.asarray(prepared.to_numpy(value), dtype=np.float64)
