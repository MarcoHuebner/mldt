"""Optional mcmlnet compatibility loader used by the external stage."""

from __future__ import annotations

import importlib.util
import itertools
import os
from pathlib import Path
from typing import Any


def load_model(model_dir: str, model_version: str, checkpoint_path: str) -> tuple[Any, ...]:
    """Load either the native mcmlnet wrapper or the documented legacy checkpoint."""
    # Resolve the configured model directory and verify both required artifacts exist.
    from mcmlnet.utils.convenience import load_trained_model

    model_dir_path = Path(model_dir).expanduser().resolve()
    os.environ.setdefault("data_dir", str(model_dir_path.parent))
    base = model_dir_path / model_version
    checkpoint = base / checkpoint_path
    config_path = base / "config_log.yaml"
    missing = [path for path in (checkpoint, config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing mcmlnet model file(s): " + ", ".join(map(str, missing)))
    # Prefer the package's own loader; use compatibility reconstruction for older installs.
    if importlib.util.find_spec("mcmlnet.training.models.uncertainty_aware_model") is not None:
        return load_trained_model(str(base), checkpoint_path)
    return _load_heteroscedastic_compatibility(config_path, checkpoint)


def _load_heteroscedastic_compatibility(
    config_path: Path, checkpoint: Path
) -> tuple[Any, Any, Any]:
    """Reconstruct the supported model from its configuration and checkpoint."""
    # Rebuild only the supported minimal heteroscedastic model from its saved config.
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    preprocessor = instantiate(config.preprocessing)
    if preprocessor.norm_1 is not None:
        preprocessor.norm_1 = torch.tensor(preprocessor.norm_1)
    if preprocessor.norm_2 is not None:
        preprocessor.norm_2 = torch.tensor(preprocessor.norm_2)
    generator = config.model.generator
    if generator.arch_type != "minimal" or int(generator.layers[-1]) != 2:
        raise ValueError("compatibility loader only supports the heteroscedastic minimal model")
    dimensions = [int(generator.n_params), *map(int, generator.layers)]
    modules = []
    for input_size, output_size in itertools.pairwise(dimensions[:-1]):
        modules.extend([torch.nn.Linear(input_size, output_size), torch.nn.LeakyReLU(0.2)])
    modules.append(torch.nn.Linear(dimensions[-2], dimensions[-1]))

    class HeteroscedasticPredictionModel(torch.nn.Module):
        def __init__(self) -> None:
            """Initialize the reconstructed feed-forward model."""
            super().__init__()
            self.model = torch.nn.Sequential(*modules)

        def predict_mean_std(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            """Return predicted means and standard deviations for input spectra."""
            raw = self.model(inputs.reshape(-1, dimensions[0]))
            mean, log_variance = raw.chunk(2, dim=-1)
            mean = torch.sigmoid(mean) if generator.sigmoid else mean
            shape = (len(inputs), -1)
            return mean.reshape(*shape), torch.sqrt(10**log_variance).reshape(*shape)

    model = HeteroscedasticPredictionModel()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(
        {
            key.removeprefix("generator."): value
            for key, value in state.items()
            if key.startswith("generator.")
        },
        strict=True,
    )
    model.eval().requires_grad_(False)
    return model, preprocessor, config
