"""Subprocess entry point for one isolated evaluation stage."""

import argparse
import json
import traceback
from pathlib import Path

from .schema import stage_document, write_json
from .stages import STAGES


def main() -> None:
    """Execute one configured stage and always serialize its outcome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    stage = config["stage"]
    try:
        # Resolve stage artifacts at the suite root for the standard directory layout.
        suite_output = (
            args.output.parent.parent if args.output.parent.name == "stages" else args.output.parent
        )
        body = STAGES[stage](config, suite_output)
        body["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - isolate and serialize stage failure
        body = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    # Always persist a structured result, including traceback details on failure.
    write_json(args.output, stage_document(stage, config, body))
    if body["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
