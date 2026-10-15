import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pyproj import CRS
from shapely.geometry import box, shape


# Output contract version of report.json/manifest.json (CLAUDE.md §6.6).
SCHEMA_VERSION = "3.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_STAMP = ".producer.json"  # Written by scripts/remote/sync_code.sh on mirrors.


class DataUnavailable(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_config(path):
    with open(path) as f:
        return (
            yaml.safe_load(f) if str(path).endswith((".yaml", ".yml")) else json.load(f)
        )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def aoi_geometry(bbox=None, geojson=None):
    if geojson:
        obj = read_config(geojson)
        if obj["type"] == "FeatureCollection":
            from shapely.ops import unary_union

            geom = unary_union([shape(f["geometry"]) for f in obj["features"]])
        else:
            geom = shape(obj.get("geometry", obj))
    else:
        if bbox is None or len(bbox) != 4:
            raise ValueError("Provide WGS84 bbox west,south,east,north or polygon")
        w, s, e, n = map(float, bbox)
        if not (-180 <= w < e <= 180 and -80 <= s < n <= 84):
            raise ValueError("Invalid WGS84 bounding box")
        geom = box(w, s, e, n)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or not geom.is_valid:
        raise ValueError("AOI must be a valid polygon")
    w, s, e, n = geom.bounds
    if not (-180 <= w < e <= 180 and -80 <= s < n <= 84):
        raise ValueError("AOI coordinates must be WGS84 within UTM latitude coverage")
    return geom


def is_metre_crs(crs):
    """Projected CRS whose horizontal axes are in metres (HR-X-008: feet are refused)."""
    if crs is None:
        return False
    crs = CRS.from_user_input(crs)
    return crs.is_projected and all(
        abs(axis.unit_conversion_factor - 1.0) < 1e-12 for axis in crs.axis_info[:2]
    )


def metric_crs(geom):
    lon, lat = geom.centroid.coords[0]
    return CRS.from_epsg(
        (32600 if lat >= 0 else 32700) + min(60, int((lon + 180) // 6) + 1)
    )


def inference_path(path):
    """Resolve an inference input; validation-only reference locations are refused."""
    candidate = Path(path).expanduser().absolute()
    resolved = candidate.resolve()
    for value in (candidate, resolved):
        for part in value.parts:
            name = part.casefold()
            if (name.startswith("reference")
                    or any(token in name for token in ("emsr927", "unosat", "gfm"))
                    or re.match(r"ems(?!r\d+(?:[-_]|$))", name)):
                raise ValueError(f"Validation-only Reference location refused as an inference input (reference data): {resolved.name}")
    return resolved


def data_root():
    return Path(os.environ.get("HIMAROAD_DATA_DIR") or "data").expanduser().resolve()


def producer_provenance(repo=REPO_ROOT):
    """Code identity of a run (HR-X-003): git checkout, else the sync stamp, else unknown."""
    from . import __version__

    record = {
        "schema_version": SCHEMA_VERSION,
        "producer_commit": "unknown",
        "producer_tree_dirty": None,
        "producer_commit_source": "unavailable",
        "package_version": __version__,
    }
    repo = Path(repo)
    if (repo / ".git").exists() and shutil.which("git"):
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True
        )
        if head.returncode == 0 and status.returncode == 0:
            record.update(
                producer_commit=head.stdout.strip(),
                producer_tree_dirty=bool(status.stdout.strip()),
                producer_commit_source="git",
            )
    elif (repo / PRODUCER_STAMP).exists():
        stamp = read_config(repo / PRODUCER_STAMP)
        record.update(
            producer_commit=stamp["commit"],
            producer_tree_dirty=stamp["dirty"],
            producer_commit_source="sync_stamp",
        )
    if record["producer_commit"] != "unknown" and not re.fullmatch(
        r"[0-9a-f]{40}", record["producer_commit"]
    ):
        raise ValueError("producer commit must be a full git SHA")
    return record
