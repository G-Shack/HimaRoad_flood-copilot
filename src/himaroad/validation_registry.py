"""Persist test-event consumption before reference access, independently of run paths."""

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .common import REPO_ROOT, checksum, data_root, now

SCHEMA = "himaroad-test-registry-v1"
CONFIG_PATH = REPO_ROOT / "configs/model_test_registry.json"


def committed_config():
    relative = CONFIG_PATH.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
                            capture_output=True, check=False)
    if result.returncode or result.stdout != CONFIG_PATH.read_bytes():
        raise ValueError("Test registry must be committed and identical to git HEAD before real test use")
    config = json.loads(result.stdout)
    if (config.get("schema_version") != SCHEMA or not isinstance(config.get("events"), dict)
            or any(item.get("status") not in {"eligible", "frozen", "consumed"} for item in config["events"].values())):
        raise ValueError("Invalid committed test registry")
    return config


def model_identity(receipt):
    cfg = receipt["model_config"]
    first = receipt["cases"][0]["model_metadata"]
    identity = {"checkpoint_sha256": receipt["checkpoint_sha256"],
                "protocol_commit": first["protocol_commit"], "model_version": first["model_version"]}
    if (not re.fullmatch(r"[0-9a-f]{64}", str(identity["checkpoint_sha256"]))
            or not re.fullmatch(r"[0-9a-f]{40}", str(identity["protocol_commit"]))
            or not isinstance(identity["model_version"], str) or not identity["model_version"]
            or cfg.get("checkpoint_sha256") != identity["checkpoint_sha256"]
            or cfg.get("protocol_commit") != identity["protocol_commit"]
            or any(any(case["model_metadata"].get(key) != value for key, value in identity.items())
                   for case in receipt["cases"])):
        raise ValueError("Frozen model identity is incomplete or inconsistent")
    return identity


def _events(receipt):
    if not isinstance(receipt.get("synthetic"), bool):
        raise ValueError("Registry use requires an explicit synthetic flag")
    events = [case["event_id"] for case in receipt["cases"]]
    if not events or len(set(events)) != len(events):
        raise ValueError("Registry use requires unique nonempty event IDs")
    return events


@contextmanager
def _locked(config):
    root = data_root() / "models/himaroad/validation"
    if root.is_symlink() or not root.resolve().is_relative_to(data_root()):
        raise ValueError("Test registry state must remain inside HIMAROAD_DATA_DIR")
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / "registry.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "test_registry.json"
        if path.is_symlink():
            raise ValueError("Test registry state cannot be a symlink")
        state = json.loads(path.read_text()) if path.exists() else {"schema_version": SCHEMA, "events": {}}
        if state.get("schema_version") != SCHEMA or not isinstance(state.get("events"), dict):
            raise ValueError("Invalid runtime test registry")
        for event, item in config["events"].items():
            if item["status"] == "consumed":
                state["events"].setdefault(event, dict(item))
        yield state
        with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix="registry-", delete=False) as temporary:
            try:
                json.dump(state, temporary, indent=2, allow_nan=False)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(temporary.name, path)
            finally:
                if os.path.exists(temporary.name):
                    os.unlink(temporary.name)
        directory_fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _refreeze(event, config, state, identity=None):
    """A committed re-freeze authorisation (amendment A2) for an unevaluated frozen event, or None.

    History is preserved: the superseded freeze stays recorded in committed and runtime registries.
    """
    committed = config["events"].get(event, {})
    authorisation = committed.get("refreeze")
    if not authorisation:
        return None
    superseded = authorisation.get("supersedes_freeze_sha256")
    runtime = state["events"].get(event)
    if (committed.get("status") != "frozen" or not authorisation.get("amendment")
            or superseded != committed.get("freeze_sha256")
            or (identity is not None and committed.get("model") != identity)):
        raise ValueError(f"Re-freeze authorisation does not match the committed frozen pin: {event}")
    if runtime is not None and (runtime.get("status") != "frozen" or runtime.get("freeze_sha256") != superseded
                                or runtime.get("model") != committed.get("model") or runtime.get("evaluation_started_at")):
        raise ValueError(f"Re-freeze requires the superseded freeze to be unevaluated: {event}")
    return authorisation


def _available(events, config, state):
    # A frozen event is never available here, even with a re-freeze authorisation (HR-X-010):
    # only register_refreeze, after the bit-identity comparison, may supersede a freeze.
    for event in events:
        if (config["events"].get(event, {}).get("status") != "eligible"
                or event in state["events"]):
            raise ValueError(f"Test event is already frozen, consumed or unregistered: {event}")


def assert_available(receipt):
    events = _events(receipt)
    if receipt["synthetic"]:
        return
    config = committed_config()
    with _locked(config) as state:
        _available(events, config, state)


def register_freeze(receipt, freeze_path):
    events = _events(receipt)
    if receipt["synthetic"]:
        return
    identity, freeze_hash = model_identity(receipt), checksum(freeze_path)
    config = committed_config()
    with _locked(config) as state:
        _available(events, config, state)
        for event in events:
            state["events"][event] = {"status": "frozen", "freeze_sha256": freeze_hash,
                                      "model": identity, "frozen_at": now()}


def _unused_frozen(events, config, state, freeze_hash, identity):
    for event in events:
        for record in (config["events"].get(event, {}), state["events"].get(event, {})):
            if (record.get("status") != "frozen" or record.get("freeze_sha256") != freeze_hash
                    or record.get("model") != identity):
                raise ValueError(f"Real evaluation requires an unused, committed freeze/model pin: {event}")


def assert_refreeze_available(previous_freeze):
    path = Path(previous_freeze).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("synthetic") is not False:
        raise ValueError("A2 re-freeze requires a registered real freeze")
    config = committed_config()
    with _locked(config) as state:
        _unused_frozen(_events(receipt), config, state, checksum(path), model_identity(receipt))
        for event in _events(receipt):
            if not _refreeze(event, config, state, model_identity(receipt)):
                raise ValueError(f"A2 re-freeze requires a committed authorisation: {event}")


def register_refreeze(receipt, freeze_path, previous_freeze):
    old_path, new_path = Path(previous_freeze).resolve(), Path(freeze_path).resolve()
    assert_refreeze_available(old_path)
    old_hash, new_hash = checksum(old_path), checksum(new_path)
    saved_receipt = json.loads(json.dumps(receipt, default=str, allow_nan=False))
    if receipt.get("synthetic") is not False or json.loads(new_path.read_text()) != saved_receipt:
        raise ValueError("Re-freeze receipt must match the saved real receipt")
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/model/compare_freezes.py"),
                             "--old", str(old_path), "--new", str(new_path)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError("A2 re-freeze is not bit-identical; registry unchanged: " + result.stdout + result.stderr)
    if checksum(old_path) != old_hash or checksum(new_path) != new_hash:
        raise ValueError("Freeze receipt changed during comparison")
    events, identity = _events(receipt), model_identity(receipt)
    config = committed_config()
    with _locked(config) as state:
        _unused_frozen(events, config, state, old_hash, identity)
        for event in events:
            record = state["events"][event]
            authorisation = _refreeze(event, config, state, identity)
            if not authorisation:
                raise ValueError(f"A2 re-freeze requires a committed authorisation: {event}")
            history = record.setdefault("history", [])
            history.append({key: value for key, value in record.items() if key != "history"}
                           | {"superseded_at": now(), "superseded_by": authorisation["amendment"]})
            record.update(freeze_sha256=new_hash, refrozen_at=now())


def begin_evaluation(receipt, freeze_hash):
    events = _events(receipt)
    if receipt["synthetic"]:
        return
    identity = model_identity(receipt)
    config = committed_config()
    with _locked(config) as state:
        _unused_frozen(events, config, state, freeze_hash, identity)
        for event in events:
            state["events"][event].update(status="consumed", evaluation_started_at=now())
