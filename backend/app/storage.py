"""Content-addressed imports and exclusive output creation."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from backend.app.schemas import EvidenceClaim, HimaRoadRun
from backend.app.settings import Settings


def canonical_run(run: HimaRoadRun) -> bytes:
    record = run.model_dump(mode="json")
    # Newly optional provenance must not change the identities of historical runs.
    for key in ("model_metadata", "component_synthetic"):
        if record[key] is None:
            del record[key]
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def output_root(settings: Settings) -> Path:
    root = (settings.data_root / "outputs").resolve()
    if not root.is_relative_to(settings.data_root) or "reference" in root.parts:
        raise ValueError("output root must remain within DATA_ROOT and outside reference")
    root.mkdir(parents=True, exist_ok=True)
    return root


def record_run(run: HimaRoadRun, settings: Settings) -> tuple[str, Path]:
    payload = canonical_run(run)
    run_id = hashlib.sha256(payload).hexdigest()
    root = output_root(settings)
    database = root / "copilot.sqlite3"
    if database.is_symlink():
        raise ValueError("database must not be a symlink")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record BLOB NOT NULL)"
        )
        connection.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)", (run_id, payload))
        saved = connection.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
        if saved is None or saved[0] != payload:
            raise ValueError("existing run record does not match content hash")
    run_dir = root / run_id
    if run_dir.is_symlink():
        raise ValueError("run output directory must not be a symlink")
    run_dir.mkdir(exist_ok=True)
    return run_id, run_dir


def write_once(run_dir: Path, name: str, content: bytes) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("output name must be a filename")
    target = run_dir / name
    # O_EXCL refuses existing regular files and symlinks, including dangling symlinks.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def store_claims(run: HimaRoadRun, claims: tuple["EvidenceClaim", ...], settings: Settings) -> str:
    """Insert immutable claims and content-backed relations in one transaction."""
    from backend.app.evidence.claims import canonical, verify_claim

    identifier, _ = record_run(run, settings)
    claim_ids = sorted(c.id for c in claims)
    members = set(claim_ids)
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("duplicate claim IDs in evidence snapshot")
    snapshot = canonical({"run_id": identifier, "claim_ids": claim_ids})
    snapshot_id = hashlib.sha256(snapshot).hexdigest()
    with sqlite3.connect(output_root(settings) / "copilot.sqlite3") as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), record BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claim_sources (
                claim_id TEXT NOT NULL REFERENCES claims(id), file TEXT NOT NULL,
                field TEXT NOT NULL, feature_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                PRIMARY KEY (claim_id, file, field, feature_id)
            );
            CREATE TABLE IF NOT EXISTS evidence_sets (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), record BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_members (
                set_id TEXT NOT NULL REFERENCES evidence_sets(id),
                claim_id TEXT NOT NULL REFERENCES claims(id), PRIMARY KEY (set_id, claim_id)
            );
            CREATE TABLE IF NOT EXISTS claim_derivations (
                claim_id TEXT NOT NULL REFERENCES claims(id),
                parent_id TEXT NOT NULL REFERENCES claims(id), PRIMARY KEY (claim_id, parent_id)
            );
        """)
        for claim in claims:
            verify_claim(claim)
            if claim.run_id != identifier:
                raise ValueError("claim belongs to another run")
            for ref in claim.sources:
                if run.files.get(ref.file) != ref.sha256:
                    raise ValueError("claim source is not backed by run file hash")
            payload = canonical(claim.model_dump(mode="json"))
            connection.execute(
                "INSERT OR IGNORE INTO claims VALUES (?, ?, ?)", (claim.id, identifier, payload)
            )
            saved = connection.execute(
                "SELECT record FROM claims WHERE id=?", (claim.id,)
            ).fetchone()
            if saved is None or saved[0] != payload:
                raise ValueError("immutable claim collision")
            for ref in claim.sources:
                connection.execute(
                    "INSERT OR IGNORE INTO claim_sources VALUES (?, ?, ?, ?, ?)",
                    (claim.id, ref.file, ref.field, ref.feature_id or "", ref.sha256),
                )
        for claim in claims:
            for parent in claim.parameters.get("derived_from", []):
                row = connection.execute(
                    "SELECT run_id FROM claims WHERE id=?", (parent,)
                ).fetchone()
                if (
                    row is None
                    or row[0] != identifier
                    or parent == claim.id
                    or parent not in members
                ):
                    raise ValueError("derivation parent must be a different claim in this run")
                connection.execute(
                    "INSERT OR IGNORE INTO claim_derivations VALUES (?, ?)", (claim.id, parent)
                )
        connection.execute(
            "INSERT OR IGNORE INTO evidence_sets VALUES (?, ?, ?)",
            (snapshot_id, identifier, snapshot),
        )
        saved = connection.execute(
            "SELECT record FROM evidence_sets WHERE id=?", (snapshot_id,)
        ).fetchone()
        if saved is None or saved[0] != snapshot:
            raise ValueError("immutable evidence snapshot collision")
        connection.executemany(
            "INSERT OR IGNORE INTO evidence_members VALUES (?, ?)",
            ((snapshot_id, c) for c in claim_ids),
        )
    return snapshot_id


def load_claims(identifier: str, settings: Settings) -> tuple["EvidenceClaim", ...]:
    """Load one immutable evidence snapshot; never combine analysis revisions."""
    from backend.app.evidence.claims import verify_claim
    from backend.app.schemas import EvidenceClaim

    database = output_root(settings) / "copilot.sqlite3"
    if database.is_symlink() or not database.exists():
        raise ValueError("evidence database unavailable")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        saved = connection.execute(
            "SELECT record FROM evidence_sets WHERE id=?", (identifier,)
        ).fetchone()
        if saved is None or hashlib.sha256(saved[0]).hexdigest() != identifier:
            raise ValueError("unknown or damaged evidence snapshot")
        snapshot = json.loads(saved[0])
        rows = connection.execute(
            "SELECT c.record FROM claims c JOIN evidence_members m ON c.id=m.claim_id "
            "WHERE m.set_id=? ORDER BY m.claim_id",
            (identifier,),
        )
        # Stream records in the covering membership-index order; do not retain
        # multi-gigabyte raw BLOBs alongside parsed claims or sort their contents.
        claims = tuple(EvidenceClaim.model_validate_json(row[0]) for row in rows)
    for claim in claims:
        verify_claim(claim)
        if claim.run_id != snapshot["run_id"]:
            raise ValueError("stored claim run mismatch")
    if [c.id for c in claims] != snapshot["claim_ids"]:
        raise ValueError("evidence snapshot membership mismatch")
    return claims


def import_evidence(path: Path, settings: Settings) -> tuple[str, tuple["EvidenceClaim", ...]]:
    from backend.app.evidence.pipeline import assemble_evidence
    from backend.app.upstream.run_dir import load_run

    run = load_run(path, settings)
    claims = assemble_evidence(run, settings)
    return store_claims(run, claims, settings), claims


def load_run_record(run_id: str, settings: Settings) -> HimaRoadRun:
    """Return the immutable imported run, without rereading mutable upstream files."""
    database = output_root(settings) / "copilot.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise ValueError("evidence database unavailable")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT record FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None or hashlib.sha256(row[0]).hexdigest() != run_id:
        raise ValueError("unknown or damaged run record")
    return HimaRoadRun.model_validate_json(row[0])
