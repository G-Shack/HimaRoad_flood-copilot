"""Sequential bounded preparation avoids concurrent SQLite writers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from backend.app.preparation import prepare
from backend.app.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings()  # pyright: ignore[reportCallIssue]
    for path in args.paths:
        start = perf_counter()
        print(path.name, "preparation started", flush=True)
        result = prepare(path, settings)
        result["seconds"] = perf_counter() - start
        (args.output / (path.name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(path.name, "passed", result["claim_count"], result["seconds"], flush=True)
