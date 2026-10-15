"""File-only preflight shared by interactive loading and bounded preparation."""

import os
import stat
from pathlib import Path

from backend.app.settings import Settings
from backend.app.upstream import contract
from backend.app.upstream.paths import RunContractError, authorised_path

MAX_RUN_BYTES = 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
MAX_PREPARED_VECTOR_BYTES = 384 * 1024**2
MAX_MEMBERS = 10_000
LARGE_VECTORS = {
    name + ".geojson" for name in (*contract.VECTORS, "possible_debris", "routes", "flowpaths")
}


def preflight(
    path: Path, config: Settings, *, prepared: bool = False, json_limit: int = MAX_JSON_BYTES
) -> Path:
    if ".." in path.parts:
        raise RunContractError("path traversal refused")
    candidate = path if path.is_absolute() else config.himaroad_data_dir / path
    root = authorised_path(candidate, config)
    for part in (candidate, *candidate.parents):
        if part == config.himaroad_data_dir.parent:
            break
        if part.is_symlink():
            raise RunContractError("run symlink refused")
    if not root.is_dir():
        raise RunContractError("run directory unavailable")
    total = count = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            member = Path(folder) / name
            authorised_path(member, config, root=root)
            count += 1
            if count > MAX_MEMBERS:
                raise RunContractError("run member limit exceeded")
            info = member.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise RunContractError("run special file refused")
            if name.casefold() in {"metrics.json", "comparison.md", "reference.json"}:
                raise RunContractError("evaluation file refused")
            total += info.st_size
            large_vector = (
                prepared
                and member.parent == root
                and (
                    name in LARGE_VECTORS or name.removesuffix(".kml") + ".geojson" in LARGE_VECTORS
                )
            )
            limit = (
                MAX_PREPARED_VECTOR_BYTES
                if large_vector
                else (
                    MAX_FILE_BYTES
                    if member.suffix.casefold() not in {".json", ".geojson"}
                    else json_limit
                )
            )
            if info.st_size > limit or total > MAX_RUN_BYTES:
                raise RunContractError("run size limit exceeded")
    return root
