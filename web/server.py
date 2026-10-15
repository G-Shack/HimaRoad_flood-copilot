"""One local app on one port: the web dashboard, the map data and the copilot API.

Start: uvicorn web.server:app --host 127.0.0.1 --port 8000 (scripts/start.sh does this).
The copilot routes (/api/runs, /api/ask, /api/report, ...) come unchanged from backend.app.api.
The /api/map routes read the saved run bundles and send slim layers to the browser.
"""

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.api import create_app
from backend.app.copilot.offline import offline_menu

WEB = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HIMAROAD_DATA_DIR", WEB.parent / "data")).resolve()
BUNDLES = DATA / "bundles"
FOOT = {"path", "footway", "steps", "pedestrian", "bridleway", "cycleway"}
HAZARDS = ("disturbance", "possible_debris", "flood_extent", "unobservable", "optical_water")
CATEGORY = ("observed_change", "likely_impact", "possible_impact", "unobservable", "no_detected_change")

app = create_app()
app.mount("/static", StaticFiles(directory=WEB), name="static")


def run_dir(run: str) -> Path:
    if run not in runs()["runs"]:
        raise HTTPException(404, "unknown_run")
    return BUNDLES / run


def read(run: str, name: str) -> dict:
    return json.loads((run_dir(run) / name).read_text())


def rounded(geometry: dict) -> dict:
    def walk(c):
        return [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], float) else [walk(x) for x in c]

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def collection(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/map/runs")
def runs() -> dict:
    names = sorted(
        p.parent.name
        for p in BUNDLES.glob("*/report.json")
        if not p.parent.is_symlink() and (p.parent / "manifest.json").is_file()
    )
    return {"runs": names}


@app.get("/api/map/menu")
def menu(language: str = "en") -> dict:
    if language not in {"en", "ne"}:
        raise HTTPException(422, "language")
    return offline_menu(language).model_dump()


@app.get("/api/map/{run}/summary")
def summary(run: str) -> dict:
    report = read(run, "report.json")
    manifest = read(run, "manifest.json")
    post = manifest.get("post") or {}
    osm = manifest.get("osm") or {}
    model = manifest.get("road_segmentation") or {}
    keys = ("schema_version", "producer_commit", "status", "event_date", "bbox", "detector",
            "observation_timestamp", "observation_age_hours", "limitations", "scenarios",
            "primary_scenario", "summary", "thresholds", "sar_geometry", "evidence_categories",
            "temporal_offsets", "timings_seconds")
    return {
        **{k: report.get(k) for k in keys},
        "run": run,
        "model": report.get("model") or manifest.get("model"),
        "pre_scene_ids": [s.get("scene_id") for s in manifest.get("pre") or []],
        "post_scene_id": post.get("scene_id"),
        "post_time_utc": post.get("timestamp"),
        "orbit": {k: post.get(k) for k in ("relative_orbit", "pass")},
        "optical_post_scene_id": (manifest.get("optical_post") or {}).get("scene_id"),
        "osm_snapshot": osm.get("snapshot"),
        "osm_cutoff_proof_passed": (osm.get("cutoff_proof") or {}).get("passed"),
        "road_segmentation": bool(model),
        "flowpaths": (run_dir(run) / "flowpaths.geojson").is_file(),
    }


@app.get("/api/map/{run}/settlements")
@lru_cache(maxsize=8)
def settlements(run: str) -> dict:
    keep = ("id", "name", "kind", "building_count", "baseline_accessible", "pedestrian_accessible",
            "isolation_confidence", "cut_in_scenarios", "depends_only_on_unknown", "category",
            "evidence_category", "potentially_exposed_buildings", "nearest_destination_tier",
            "nearest_destination_cost_m", "affected_connections",
            *(f"scenario{n}_accessible" for n in range(1, 5)))
    features = [
        {"type": "Feature", "geometry": rounded(f["geometry"]),
         "properties": {k: f["properties"].get(k) for k in keep}}
        for f in read(run, "settlements.geojson")["features"]
    ]
    return collection(features)


@app.get("/api/map/{run}/roads")
@lru_cache(maxsize=4)
def roads(run: str) -> Response:
    # Vehicle roads in full; footpaths only when they carry evidence of change.
    features = []
    for f in read(run, "roads.geojson")["features"]:
        p = f["properties"]
        cat = p.get("evidence_category") or "no_detected_change"
        if p.get("highway") in FOOT and cat in ("no_detected_change", "unobservable"):
            continue
        features.append({"type": "Feature", "geometry": rounded(f["geometry"]),
                         "properties": {"id": p.get("id"), "highway": p.get("highway"), "c": cat,
                                        "name": p.get("name")}})
    return Response(json.dumps(collection(features), separators=(",", ":")),
                    media_type="application/json")


@app.get("/api/map/{run}/buildings")
@lru_cache(maxsize=8)
def buildings(run: str) -> dict:
    features = [
        {"type": "Feature", "geometry": rounded(f["geometry"]),
         "properties": {"id": f["properties"].get("id"), "c": f["properties"].get("evidence_category")}}
        for f in read(run, "buildings.geojson")["features"]
        if f["properties"].get("potentially_exposed")
    ]
    return collection(features)


@app.get("/api/map/{run}/layer/{name}")
@lru_cache(maxsize=32)
def layer(run: str, name: str) -> dict:
    if name not in (*HAZARDS, "destinations", "flowpaths"):
        raise HTTPException(404, "unknown_layer")
    path = run_dir(run) / f"{name}.geojson"
    if not path.is_file():
        return collection([])
    features = [
        {"type": "Feature", "geometry": rounded(f["geometry"]), "properties": {
            k: v for k, v in f["properties"].items()
            if k in ("id", "kind", "source_name", "length_m", "stop_reason", "corridor_m",
                     "destination_tier", "amenity", "name", "label")}}
        for f in json.loads(path.read_text())["features"] if f.get("geometry")
    ]
    return collection(features)


@lru_cache(maxsize=8)
def radar_png(run: str, which: str) -> tuple[bytes, list]:
    import rasterio
    from PIL import Image
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(run_dir(run) / f"{which}.tif") as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        out = np.full((height, width), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), out, dst_transform=transform, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    values = out[np.isfinite(out)]
    if values.size and np.nanmedian(values) > 0 and np.nanmax(values) < 10:
        out = 10 * np.log10(np.clip(out, 1e-6, None))  # linear gamma0 to dB
    gray = np.clip((out + 30) / 30, 0, 1) * 255
    rgba = np.dstack([gray, gray, gray, np.where(np.isfinite(out), 255, 0)]).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, "PNG", optimize=True)
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    return buffer.getvalue(), [[south, west], [north, east]]


@app.get("/api/map/{run}/radar/{which}.png")
def radar(run: str, which: str) -> Response:
    if which not in {"pre", "post"}:
        raise HTTPException(404, "unknown_image")
    return Response(radar_png(run, which)[0], media_type="image/png")


@app.get("/api/map/{run}/radar/{which}.json")
def radar_bounds(run: str, which: str) -> dict:
    if which not in {"pre", "post"}:
        raise HTTPException(404, "unknown_image")
    return {"bounds": radar_png(run, which)[1]}


class Point(BaseModel):
    lon: float
    lat: float


@app.post("/api/map/{run}/flowpath")
def flowpath(run: str, point: Point) -> dict:
    from himaroad.flowpath import trace_for_run

    try:
        return trace_for_run(run_dir(run), point.lon, point.lat)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from None
