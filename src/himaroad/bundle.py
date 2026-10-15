"""Portable run bundles (HR-X-005): relative paths, checksums and referenced inputs."""

import shutil
from pathlib import Path, PurePosixPath

from .common import checksum, inference_path, now, read_config, write_json

BUNDLE_FORMAT = "himaroad-bundle-1"
INDEX = "bundle.json"
# Evaluation outputs never travel with evidence bundles (reference firewall).
EVALUATION_FILES = {"metrics.json", "comparison.md", "reference.json"}
OSM_LAYERS = ("roads", "buildings", "settlements", "destinations")


def _relativise(value, mapping, where):
    """Replace every absolute path string; fail on any path the bundle does not hold."""
    if isinstance(value, dict):
        return {k: _relativise(v, mapping, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_relativise(v, mapping, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str) and value.startswith("/"):
        if value not in mapping:
            raise ValueError(f"{where}: path {Path(value).name} is not part of the bundle")
        return mapping[value]
    return value


def create_bundle(run, out):
    """Copy a run and its checksummed inputs into a self-contained directory."""
    run, out = Path(run).resolve(), Path(out).resolve()
    if "reference" in out.parts:
        raise ValueError("Bundles must not live under a reference path")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Bundle directory must be new or empty")
    report = read_config(run / "report.json")
    manifest = read_config(run / "manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    files, mapping, rewrites, excluded = {}, {}, [], []
    for source in sorted(run.iterdir()):
        if source.is_symlink():
            raise ValueError(f"Symlink in run output refused: {source.name}")
        if source.is_dir():
            raise ValueError(f"Unexpected directory in run output: {source.name}")
        mapping[str(source)] = source.name
        if source.name in EVALUATION_FILES:
            excluded.append(source.name)
        elif source.name not in {"report.json", "manifest.json"}:
            shutil.copy2(source, out / source.name)
            files[source.name] = checksum(out / source.name)
    if manifest.get("polygon"):
        polygon = inference_path(manifest["polygon"])
        manifest["inputs"].append(
            {"path": str(polygon), "sha256": checksum(polygon), "source": "AOI polygon"}
        )
        rewrites.append("polygon added to inputs with its checksum")
    (out / "inputs").mkdir()
    for number, item in enumerate(manifest.get("inputs", [])):
        if item["path"] in mapping:
            continue
        source = inference_path(item["path"])  # also refuses reference locations
        target = f"inputs/{number:02d}-{source.name}"
        shutil.copy2(source, out / target)
        digest = checksum(out / target)
        if digest != item["sha256"]:
            raise ValueError(f"Input {source.name} changed since the run (checksum mismatch)")
        mapping[item["path"]] = target
        files[target] = digest
    if isinstance(manifest.get("processed_cache"), dict):
        cache = manifest["processed_cache"]
        manifest["processed_cache"] = {
            "file": Path(cache["path"]).name,
            "hit": cache.get("hit"),
            "bundled": False,
        }
        rewrites.append("processed_cache.path replaced by its file name (derived cache, not bundled)")
    configuration = manifest.get("configuration") or {}
    if configuration.get("processed_cache_dir"):
        configuration["processed_cache_dir"] = Path(configuration["processed_cache_dir"]).name
        rewrites.append("configuration.processed_cache_dir reduced to its name (not bundled)")
    manifest = _relativise(manifest, mapping, "manifest")
    report = _relativise(report, mapping, "report")
    rewrites.append("absolute paths replaced by paths relative to the bundle root")
    for name, value in (("manifest.json", manifest), ("report.json", report)):
        write_json(out / name, value)
        files[name] = checksum(out / name)
    write_json(
        out / INDEX,
        {
            "format": BUNDLE_FORMAT,
            "created_utc": now(),
            "source_run": run.name,
            "schema_version": report.get("schema_version"),
            "producer_commit": report.get("producer_commit"),
            "synthetic": report.get("synthetic"),
            "files": files,
            "manifest_rewrites": rewrites,
            "excluded_evaluation_files": excluded,
        },
    )
    verify_bundle(out)
    return out


def _safe_member(root, name):
    """Reject an index name before anything is read (HR-X-006): it must be a plain
    relative path inside the bundle, with no symlink in any component."""
    parts = PurePosixPath(name).parts
    if (
        not parts
        or PurePosixPath(name).is_absolute()
        or "\\" in name
        or any(part in ("", ".", "..") for part in parts)
        # Case-folded: macOS and Windows open "METRICS.JSON" as "metrics.json" (review L1).
        or "reference" in [part.casefold() for part in parts]
        or parts[-1].casefold() in EVALUATION_FILES
    ):
        raise ValueError(f"Unsafe bundle member name: {name!r}")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in bundle member: {name}")
    if not current.is_file():
        raise ValueError(f"Listed bundle member missing: {name}")
    return current


def _path_records(value):
    """Every dict with a string "path" key, at any depth."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield value
        for item in value.values():
            yield from _path_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _path_records(item)


def verify_bundle(path):
    """Inventory, containment and symlink checks first; checksums only afterwards."""
    root = Path(path).resolve()
    index_path = root / INDEX
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("Not a HimaRoad bundle")
    index = read_config(index_path)
    if index.get("format") != BUNDLE_FORMAT or not isinstance(index.get("files"), dict):
        raise ValueError("Not a HimaRoad bundle")
    members = {name: _safe_member(root, name) for name in index["files"]}
    present = set()
    for item in root.rglob("*"):
        name = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ValueError(f"Symlink in bundle: {name}")
        if item.is_file():
            present.add(name)
        elif not item.is_dir():
            raise ValueError(f"Special file in bundle: {name}")  # FIFO, socket, device (review L2)
    unlisted = present - set(members) - {INDEX}
    if unlisted:
        raise ValueError(f"Files not listed in {INDEX}: {sorted(unlisted)}")
    if any(Path(name).name.casefold() in EVALUATION_FILES for name in present):
        raise ValueError("Evaluation files must not travel in a bundle")
    manifest, report = read_config(members["manifest.json"]), read_config(members["report.json"])
    _relativise(manifest, {}, "manifest")
    _relativise(report, {}, "report")
    inputs = {item["path"]: item.get("sha256") for item in manifest.get("inputs", [])}
    osm = manifest.get("osm", {})
    declared = list(inputs) + [osm[key] for key in OSM_LAYERS if isinstance(osm.get(key), str)]
    for value in declared + list(report.get("exports", {}).values()):
        if value not in members:
            raise ValueError(f"Declared path {value} is not a listed bundle member")
    # Every record with a "path" anywhere in the manifest (scene, masks, dem, hand,
    # optical) must be a checksummed input inside the bundle (review M4).
    for record in _path_records(manifest):
        if record["path"] not in inputs or record["path"] not in members:
            raise ValueError(f"Manifest path {record['path']} is not a checksummed bundle input")
        if record.get("sha256") not in (None, inputs[record["path"]]):
            raise ValueError(f"Conflicting checksum for {record['path']}")
    for name, digest in index["files"].items():
        if checksum(members[name]) != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return {"bundle": str(root), "files": len(index["files"]), "status": "verified"}
