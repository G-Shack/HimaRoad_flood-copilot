"""Import a run without starting an LLM or reading reference material."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from backend.app.evidence.queries import get_evidence_validation
from backend.app.settings import Settings
from backend.app.storage import import_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    settings = Settings()  # pyright: ignore[reportCallIssue]
    identifier, claims = import_evidence(args.run, settings)
    integrity = get_evidence_validation(claims).claims[0].value
    print(
        json.dumps(
            {
                "evidence_id": identifier,
                "run_id": claims[0].run_id,
                "claims": len(claims),
                "seconds": perf_counter() - started,
                "synthetic": any("synthetic" in c.quality_flags for c in claims),
                "evidence_integrity": integrity,
            },
            sort_keys=True,
        )
    )
    if integrity["result"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
