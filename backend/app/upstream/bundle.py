"""Strict evidence-bundle boundary, independent of producer verification."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from backend.app.settings import Settings
from backend.app.upstream.paths import RunContractError, authorised_path, check_sources

FORBIDDEN = {"metrics.json", "comparison.md", "reference.json"}


def verify_bundle(root: Path, settings: Settings) -> dict:
    index_path = root / "bundle.json"
    if index_path.is_symlink():
        raise RunContractError("bundle index symlink refused")
    try:
        index = json.loads(index_path.read_text())
    except (OSError, ValueError) as exc:
        raise RunContractError("bundle.json: unreadable index") from exc
    if not isinstance(index, dict) or index.get("format") != "himaroad-bundle-1":
        raise RunContractError("bundle.json: unsupported format")
    files = index.get("files")
    if not isinstance(files, dict) or not {"report.json", "manifest.json"} <= files.keys():
        raise RunContractError("bundle.json: missing file inventory")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p.casefold() in {"", "..", ".", "reference"} for p in name.split("/"))
            or PurePosixPath(name).name.casefold() in FORBIDDEN
            or name.casefold() == "bundle.json"
        ):
            raise RunContractError("bundle.json: forbidden member path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise RunContractError("bundle.json: invalid checksum")
    if len({name.casefold() for name in files}) != len(files):
        raise RunContractError("bundle.json: case-colliding member paths")
    present = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RunContractError("bundle symlink refused")
        if path.is_file():
            present.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise RunContractError("bundle special file refused")
    if present != set(files) | {"bundle.json"}:
        raise RunContractError("bundle inventory: unlisted or missing files")
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    check_sources(manifest)
    check_sources(report)
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        for item in inputs
    ):
        raise RunContractError("bundle manifest: invalid input inventory")
    declared = {item["path"]: item["sha256"] for item in inputs}
    if len(declared) != len(inputs) or any(
        files.get(p) != digest for p, digest in declared.items()
    ):
        raise RunContractError("bundle manifest: conflicting or unlisted input checksum")

    def check_records(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and (
                value["path"] not in declared
                or value.get("sha256") not in (None, declared[value["path"]])
            ):
                raise RunContractError("bundle manifest: path is not a checksummed input")
            for item in value.values():
                check_records(item)
        elif isinstance(value, list):
            for item in value:
                check_records(item)

    check_records(manifest)
    for name, digest in files.items():
        path = authorised_path(root / name, settings, root=root)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise RunContractError(f"bundle checksum mismatch: {name}")

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path" and isinstance(item, str):
                    authorised_path(root / item, settings, root=root)
                paths(item)
        elif isinstance(value, list):
            for item in value:
                paths(item)
        elif isinstance(value, str) and value.startswith("/"):
            raise RunContractError("bundle contains an absolute path")

    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    paths(manifest)
    paths(report)
    for item in [
        *report.get("exports", {}).values(),
        *(
            manifest.get("osm", {}).get(k)
            for k in ("roads", "buildings", "settlements", "destinations")
        ),
        manifest.get("polygon"),
    ]:
        if isinstance(item, str):
            authorised_path(root / item, settings, root=root)
    return index
