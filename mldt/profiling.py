"""Small, reusable cProfile helpers for tutorial and research scripts.

The helpers work with plain dictionaries and Hydra/OmegaConf configs. They keep
profiling optional so the same entrypoint can be used for quick tests and for
instrumented runs.
"""

from __future__ import annotations

import cProfile
import inspect
import io
import pstats
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeAlias, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
Config: TypeAlias = Mapping[str, Any]


def render_cprofile_report(
    profiler: cProfile.Profile,
    *,
    sort_by: str,
    top_n: int,
) -> str:
    """Render a trimmed text summary from a populated cProfile profiler."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)
    return stream.getvalue()


def write_profile_artifacts(
    *,
    output_prefix: str | None,
    cprofile_report: str,
    profiler: cProfile.Profile,
    forward_report: str | None = None,
) -> None:
    """Persist cProfile and optional forward-model profiling artifacts."""
    if not output_prefix:
        return

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(prefix.with_suffix(".prof")))
    prefix.with_suffix(".txt").write_text(cprofile_report)
    if forward_report:
        prefix.with_name(f"{prefix.name}_forward").with_suffix(".txt").write_text(forward_report)


def run_callable_with_optional_profiling(
    fn: Callable[[], T],
    *,
    enabled: bool,
    sort_by: str,
    top_n: int,
    output_prefix: str | None,
    summary_label: str,
    forward_report_fn: Callable[[], str | None] | None = None,
) -> T:
    """Run a callable and optionally capture cProfile plus forward-model timings."""
    if not enabled:
        return fn()

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = fn()
    finally:
        profiler.disable()

    cprofile_report = render_cprofile_report(
        profiler,
        sort_by=sort_by,
        top_n=top_n,
    )
    print(summary_label)
    print(cprofile_report)

    forward_report = None if forward_report_fn is None else forward_report_fn()
    if forward_report:
        print(forward_report)

    write_profile_artifacts(
        output_prefix=output_prefix,
        cprofile_report=cprofile_report,
        forward_report=forward_report,
        profiler=profiler,
    )
    return result


def _as_mapping(value: Any, *, field_name: str) -> Config:
    """Return a mapping from plain dict-like and OmegaConf config values."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value

    # Avoid a hard dependency on OmegaConf. Hydra users can pass DictConfig
    # values and we convert them when OmegaConf is available in their project.
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise TypeError(
            f"`{field_name}` must be a mapping when OmegaConf is not installed."
        ) from exc

    converted = OmegaConf.to_container(value, resolve=True)
    if not isinstance(converted, Mapping):
        raise TypeError(f"`{field_name}` must be a mapping when configured.")
    return converted


def profiling_config(cfg: Config) -> Config:
    """Return the optional profiling config as a mapping."""
    return _as_mapping(cfg.get("profiling"), field_name="profiling")


def profiling_enabled(cfg: Config) -> bool:
    """Return whether cProfile is enabled in the config."""
    profiling_cfg = profiling_config(cfg)
    return bool(profiling_cfg.get("enabled", False))


def run_with_optional_cprofile(
    cfg: Config,
    *,
    output_dir: Path,
    output_name: str,
    fn: Callable[[], T],
) -> T:
    """Run `fn`, optionally persisting cProfile artifacts from Hydra config."""
    if not profiling_enabled(cfg):
        return fn()

    profiling_cfg = profiling_config(cfg)
    sort_by = str(profiling_cfg.get("sort_by", "cumtime"))
    top_n = int(profiling_cfg.get("top_n", 80))
    output_subdir = str(profiling_cfg.get("output_subdir", "profiles"))

    profile_dir = output_dir / output_subdir
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(profile_dir / output_name)

    return run_callable_with_optional_profiling(
        fn,
        enabled=True,
        sort_by=sort_by,
        top_n=top_n,
        output_prefix=output_prefix,
        summary_label=f"{output_name} cProfile summary:",
    )


def with_optional_cprofile(
    *,
    output_name: str,
    output_dir_arg: str = "run_dir",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate an entrypoint with optional cProfile profiling.

    The wrapped function must receive a ``cfg`` mapping and an output directory
    argument named by ``output_dir_arg``.
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        """Wrap an entrypoint with optional profiling."""
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Run the entrypoint with profiling when configured."""
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            if "cfg" not in bound_args.arguments:
                raise TypeError(f"`{fn.__name__}` must accept a `cfg` argument for profiling.")
            if output_dir_arg not in bound_args.arguments:
                raise TypeError(
                    f"`{fn.__name__}` must accept a `{output_dir_arg}` argument "
                    "for profiling output."
                )

            return run_with_optional_cprofile(
                bound_args.arguments["cfg"],
                output_dir=Path(bound_args.arguments[output_dir_arg]),
                output_name=output_name,
                fn=lambda: fn(*args, **kwargs),
            )

        return wrapper

    return decorator
