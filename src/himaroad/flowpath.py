"""DEM-derived flow path from a point (bonus, A-D10/D28): not an inundation forecast.

The Copernicus DEM is conditioned with priority-flood filling (Barnes et al. 2014,
via ``hand.condition``), so every inland cell has a strictly lower neighbour; the
path follows D8 steepest descent (``hand.d8_receivers``) from the cell containing
the start point until a cell has no lower neighbour. That cell is on the grid
edge (``grid_edge``), next to nodata (``nodata``) or, which conditioning should
prevent, an interior cell (``outlet``). The path says where water from the point
would go on the DEM surface; it says nothing about depth, timing or extent.
"""

import json
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from shapely.geometry import LineString, mapping

from .common import is_metre_crs
from shapely.ops import nearest_points

from himaroad.hand import condition, d8_receivers

LABEL = "DEM-derived flow path, not an inundation forecast"
METHOD = "D8 steepest descent on priority-flood conditioned DEM (Barnes 2014)"
DEFAULT_CORRIDOR_M = 500.0


@lru_cache(maxsize=2)
def _surface(path: str, mtime_ns: int, size: int):
    """Conditioned receivers of a DEM file (cached per file version within a process)."""
    with rasterio.open(path) as src:
        if not is_metre_crs(src.crs):
            raise ValueError("The flow path needs a DEM on a projected grid in metres")
        a, b, _, d, e, _ = tuple(src.transform)[:6]
        if b or d or not np.isclose(abs(a), abs(e)):
            raise ValueError("The flow path needs square, north-up DEM cells")
        dem = src.read(1, masked=True).astype("float64").filled(np.nan)
        geo = src.transform, src.crs
    filled, _, _ = condition(dem)
    receiver = d8_receivers(filled, abs(a))
    return dem, receiver, geo[0], geo[1]


def _load(dem_path):
    path = Path(dem_path).resolve()
    stat = path.stat()
    return _surface(str(path), stat.st_mtime_ns, stat.st_size)


def _stop_reason(dem, index):
    rows, cols = dem.shape
    r, c = divmod(int(index), cols)
    if r in (0, rows - 1) or c in (0, cols - 1):
        return "grid_edge"
    if not np.isfinite(dem[r - 1 : r + 2, c - 1 : c + 2]).all():
        return "nodata"
    return "outlet"


def trace(dem_path, lon: float, lat: float, max_length_m: float | None = None) -> dict:
    """Follow D8 steepest descent downstream from (lon, lat) in WGS84."""
    dem, receiver, geotransform, crs = _load(dem_path)
    rows, cols = dem.shape
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    row, col = rowcol(geotransform, x, y)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"Start point ({lon}, {lat}) is outside the DEM")
    if not np.isfinite(dem[row, col]):
        raise ValueError(f"Start point ({lon}, {lat}) is on DEM nodata")
    cell = abs(geotransform.a)
    index = row * cols + col
    cells = [index]
    length = 0.0
    while True:
        nxt = receiver[index]
        if nxt < 0:
            stop = _stop_reason(dem, index)
            break
        (r0, c0), (r1, c1) = divmod(int(index), cols), divmod(int(nxt), cols)
        step = cell * float(np.hypot(r1 - r0, c1 - c0))
        if max_length_m is not None and length + step > max_length_m:
            stop = "max_length"
            break
        length += step
        index = nxt
        cells.append(index)
    r, c = np.divmod(np.asarray(cells), cols)
    xs, ys = xy(geotransform, r, c)  # cell centres
    coords = list(zip(np.asarray(xs).tolist(), np.asarray(ys).tolist()))
    if len(coords) == 1:  # the start cell is already terminal: a zero-length line
        coords = coords * 2
    return {
        "geometry_metric": LineString(coords),
        "crs": crs.to_string(),
        "length_m": length,
        "cells": len(cells),
        "stop_reason": stop,
        "start_cell": [int(row), int(col)],
        "start_snapped_m": float(np.hypot(x - coords[0][0], y - coords[0][1])),
    }


def _text(value):
    return value if isinstance(value, str) and value else None


def settlements_along(path_metric: LineString, settlements: gpd.GeoDataFrame, corridor_m: float) -> list[dict]:
    """Settlements within corridor_m of the path, in downstream order (along-path distance)."""
    if settlements.empty:
        return []
    offsets = settlements.geometry.distance(path_metric)
    records = []
    for (_, row), offset in zip(settlements.iterrows(), offsets):
        if offset > corridor_m:
            continue
        on_path = nearest_points(path_metric, row.geometry)[0]
        records.append(
            {
                "settlement_id": str(row["id"]),
                "name": _text(row.get("name")),
                "along_m": float(path_metric.project(on_path)),
                "offset_m": float(offset),
            }
        )
    return sorted(records, key=lambda item: (item["along_m"], item["settlement_id"]))


def flowpath_feature_collection(paths: list[dict]) -> dict:
    """WGS84 GeoJSON for traced paths.

    Each item is a ``trace`` result plus ``source_name``, ``source_lon``,
    ``source_lat``, ``corridor_m``, ``dem_source`` and ``settlements``
    (from ``settlements_along``).
    """
    features = []
    for n, path in enumerate(paths):
        to_wgs84 = Transformer.from_crs(path["crs"], 4326, always_xy=True).transform
        line = path["geometry_metric"]
        flow_id = f"flowpath/{n}"
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely.transform(line, to_wgs84, interleaved=False)),
                "properties": {
                    "kind": "flow_path",
                    "id": flow_id,
                    "source_name": path.get("source_name"),
                    "source_lon": path["source_lon"],
                    "source_lat": path["source_lat"],
                    "start_snapped_m": path["start_snapped_m"],
                    "length_m": path["length_m"],
                    "cells": path["cells"],
                    "stop_reason": path["stop_reason"],
                    "corridor_m": path["corridor_m"],
                    "dem_source": path["dem_source"],
                    "method": METHOD,
                    "label": LABEL,
                    "settlements": path["settlements"],
                },
            }
        )
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shapely.transform(line.buffer(path["corridor_m"]), to_wgs84, interleaved=False)),
                "properties": {
                    "kind": "corridor",
                    "id": f"{flow_id}/corridor",
                    "flow_path_id": flow_id,
                    "corridor_m": path["corridor_m"],
                    "label": LABEL,
                },
            }
        )
    # Round-trip through JSON so tuples become lists and the result is plain GeoJSON.
    return json.loads(json.dumps({"type": "FeatureCollection", "features": features}))


def trace_sources(dem_path, sources, settlements, corridor_m=DEFAULT_CORRIDOR_M, dem_source="", max_length_m=None) -> dict:
    """Trace each ``{"name", "lon", "lat"}`` source and return the flowpaths FeatureCollection.

    ``settlements`` is a GeoDataFrame in any CRS; it is projected to the DEM CRS.
    A source outside the DEM or on nodata raises ValueError.
    """
    paths = []
    for source in sources:
        path = trace(dem_path, source["lon"], source["lat"], max_length_m)
        local = settlements.to_crs(path["crs"]) if not settlements.empty else settlements
        path |= {
            "source_name": source.get("name"),
            "source_lon": float(source["lon"]),
            "source_lat": float(source["lat"]),
            "corridor_m": float(corridor_m),
            "dem_source": dem_source,
            "settlements": settlements_along(path["geometry_metric"], local, corridor_m),
        }
        paths.append(path)
    return flowpath_feature_collection(paths)


def run_dem(run_dir) -> tuple[Path, str]:
    """The run's DEM (buffered input DEM from the manifest, else the AOI-grid dem.tif) and its source text."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dem = manifest.get("dem") or {}
    candidates = []
    if dem.get("path"):
        path = Path(dem["path"])
        candidates.append(path if path.is_absolute() else run_dir / path)
    candidates.append(run_dir / "dem.tif")
    for path in candidates:
        if path.exists():
            shown = path.relative_to(run_dir) if path.is_relative_to(run_dir) else path
            return path, f"{dem.get('source', 'unknown DEM source')} [{shown}]"
    raise FileNotFoundError(f"No DEM found for run {run_dir}: tried {[str(p) for p in candidates]}")


def trace_for_run(run_dir, lon: float, lat: float, corridor_m: float = DEFAULT_CORRIDOR_M, name: str | None = None) -> dict:
    """Flow path from (lon, lat) on a run's DEM with the run's settlements (CLI core)."""
    run_dir = Path(run_dir)
    dem_path, dem_source = run_dem(run_dir)
    settlements = gpd.read_file(run_dir / "settlements.geojson")
    return trace_sources(
        dem_path, [{"name": name, "lon": lon, "lat": lat}], settlements, corridor_m, dem_source
    )
