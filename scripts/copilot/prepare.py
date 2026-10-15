"""Prepare a large saved run under the launcher before interactive loading."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from backend.app.preparation import prepare
from backend.app.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    print("preparation started: bounded validation, evidence assembly and storage", flush=True)
    result = prepare(args.run, Settings())  # pyright: ignore[reportCallIssue]
    result["seconds"] = perf_counter() - start
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "preparation complete",
        result["claim_count"],
        "claims",
        result["seconds"],
        "seconds",
        flush=True,
    )
