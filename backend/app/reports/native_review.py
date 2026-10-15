"""A native-review notice bound to the shipped inventory and its source bytes."""

import hashlib
import json
from pathlib import Path

PENDING = "not yet reviewed by a native speaker"
REVIEWED = "Nepali templates reviewed by Siku, a native speaker"


def pair_hash(record: dict) -> str:
    value = json.dumps([record["en"], record["ne"]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def review_notice() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        inventory = json.loads(Path(__file__).with_name("template-inventory.json").read_text())
        receipt = json.loads(Path(__file__).with_name("native-review.json").read_text())
        reviewed = {r["id"]: r for r in receipt["templates"]}
        if receipt["reviewer"] != "Siku" or receipt["review_date"] != "2026-10-15":
            return PENDING
        for path, digest in inventory["sources"].items():
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
                return PENDING
        if not inventory["templates"]:
            return PENDING
        for record in inventory["templates"]:
            row = reviewed.get(record["id"], {})
            if row.get("pair_sha256") != pair_hash(record):
                return PENDING
            if row.get("decision") != "approved" and row.get("resolution") not in {
                "applied",
                "technical_regex_retained",
            }:
                return PENDING
    except (OSError, ValueError, KeyError, TypeError):
        # Missing or stale review evidence must never produce a reviewed claim.
        return PENDING
    return REVIEWED
