"""Explicit large-run preparation; interactive requests cannot grant admission."""

import hashlib
import json
import sqlite3
from pathlib import Path

from backend.app.evidence.claims import canonical
from backend.app.schemas import HimaRoadRun
from backend.app.settings import Settings
from backend.app.storage import import_evidence, load_claims, load_run_record, output_root
from backend.app.upstream.budget import preflight


def software_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    # Admission follows the claim producer, separately from presentation code.
    app = root / "backend/app"
    files = [
        *(
            p
            for folder in ("analysis", "evidence", "upstream")
            for p in sorted((app / folder).glob("*.py"))
        ),
        *(app / name for name in ("schemas.py", "settings.py", "storage.py")),
        root / "upstream/himaroad.ref",
    ]
    return hashlib.sha256(
        canonical(
            {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        )
    ).hexdigest()


def fingerprint(path: Path, settings: Settings) -> str:
    root = preflight(path, settings, prepared=True)
    # Declared reference purposes/sources/paths must fail before any imagery or
    # vector hashing, including when a previously prepared source has changed.
    from backend.app.upstream.paths import check_sources
    from backend.app.upstream.run_dir import _check_paths, _json

    for name in ("report.json", "manifest.json"):
        metadata = _json(root / name)
        check_sources(metadata)
        _check_paths(metadata, root, settings)
    records = {}
    for member in sorted(root.rglob("*")):
        if member.is_file():
            with member.open("rb") as handle:
                records[member.relative_to(root).as_posix()] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    return hashlib.sha256(
        canonical({"path": str(root), "files": records, "software": software_digest()})
    ).hexdigest()


def prepare(path: Path, settings: Settings) -> dict:
    root = preflight(path, settings, prepared=True)
    before = fingerprint(root, settings)
    identifier, claims = import_evidence(root, settings)
    integrity = next(c.value for c in claims if c.kind == "evidence_validation")
    if integrity["result"] != "passed":
        raise ValueError("preparation requires passed evidence integrity")
    if fingerprint(root, settings) != before:
        raise ValueError("run changed during preparation")
    record = {
        "path": str(root),
        "fingerprint": before,
        "evidence_id": identifier,
        "run_id": claims[0].run_id,
    }
    payload = canonical(record)
    with sqlite3.connect(output_root(settings) / "copilot.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS prepared_runs "
            "(path TEXT PRIMARY KEY, record BLOB NOT NULL, digest TEXT NOT NULL)"
        )
        db.execute(
            "INSERT OR REPLACE INTO prepared_runs VALUES (?, ?, ?)",
            (str(root), payload, hashlib.sha256(payload).hexdigest()),
        )
    return {
        **record,
        "claim_count": len(claims),
        "producer_commit": claims[0].parameters["producer_commit"],
        "adapter_pin": claims[0].parameters["adapter_pin"],
    }


def prepared_run(
    path: Path, settings: Settings, *, verify_evidence: bool = True
) -> tuple[str, HimaRoadRun] | None:
    root = preflight(path, settings, prepared=True)
    database = output_root(settings) / "copilot.sqlite3"
    if database.is_symlink():
        raise ValueError("database symlink refused")
    if not database.is_file():
        return None
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        if (
            db.execute("SELECT name FROM sqlite_master WHERE name='prepared_runs'").fetchone()
            is None
        ):
            return None
        row = db.execute(
            "SELECT record, digest FROM prepared_runs WHERE path=?", (str(root),)
        ).fetchone()
    if row is None:
        return None
    if hashlib.sha256(row[0]).hexdigest() != row[1]:
        raise ValueError("damaged preparation record")
    record = json.loads(row[0])
    if record["path"] != str(root) or record["fingerprint"] != fingerprint(root, settings):
        raise ValueError("preparation stale; prepare again")
    run = load_run_record(record["run_id"], settings)
    claims = load_claims(record["evidence_id"], settings) if verify_evidence else ()
    if run.run_path != root or any(c.run_id != record["run_id"] for c in claims):
        raise ValueError("preparation snapshot mismatch")
    return record["evidence_id"], run
