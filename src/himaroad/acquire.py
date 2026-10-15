import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import rasterio

from .common import DataUnavailable, checksum, inference_path, now, read_config, utc
from .osm import check_osm_snapshot, edit_timestamp_proof

S1_PLATFORMS = ("S1A", "S1B", "S1C", "S1D")


def scene_platform(scene: dict) -> str:
    """Sentinel-1 unit "S1A".."S1D" from platform/platform_number, else "unknown"."""
    name = str(scene.get("platform") or "").upper().replace("SENTINEL-", "S")
    number = str(scene.get("platform_number") or "").upper()
    from_number = f"S1{number}" if f"S1{number}" in S1_PLATFORMS else None
    from_name = name if name in S1_PLATFORMS else None
    if from_name and from_number and from_name != from_number:
        raise ValueError(
            f"Conflicting Sentinel-1 platform metadata: {scene.get('platform')} "
            f"vs platform_number {scene.get('platform_number')}"
        )
    return from_name or from_number or "unknown"


def pairing_provenance(pre: list, post: dict) -> dict:
    """Platform per scene (pre stack in order, post last).

    Cross-platform pairs are flagged, not rejected.
    """
    platforms = [scene_platform(scene) for scene in [*pre, post]]
    known = {p for p in platforms if p != "unknown"}
    return {
        "platforms": platforms,
        "cross_platform": len(known) > 1,
        "platform_unknown": "unknown" in platforms,
    }


def pairing_limitations(pairing: dict) -> list:
    notes = []
    if pairing["cross_platform"]:
        pre = ", ".join(dict.fromkeys(pairing["platforms"][:-1]))
        notes.append(
            "Pre/post Sentinel-1 scenes come from different platforms "
            f"(pre: {pre}; post: {pairing['platforms'][-1]}); radiometric "
            "differences between units may add apparent change."
        )
    if pairing["platform_unknown"]:
        notes.append(
            "Sentinel-1 platform is unknown for at least one pre/post scene; "
            "a cross-platform pair cannot be ruled out."
        )
    return notes


def validate_pair(pre, post, event, cutoff=None):
    if not pre or not post:
        raise DataUnavailable(
            "insufficient_observations", "Pre/post SAR observations are required"
        )
    for scene in pre:
        if utc(scene["timestamp"]) >= utc(event):
            raise ValueError("Background observation must precede event")
    if utc(post["timestamp"]) <= utc(event):
        raise ValueError("Post-event observation must be strictly after event")
    for scene in [*pre, post]:
        if cutoff and utc(scene["timestamp"]) > utc(cutoff):
            raise ValueError("Observation exceeds analysis cutoff")
        for key in ("relative_orbit", "pass", "instrument_mode", "polarization"):
            if key not in scene or scene[key] != post.get(key):
                raise ValueError(f"Incompatible or missing radar {key}")
        if scene.get("native_sensor_resolution_m") != post.get(
            "native_sensor_resolution_m"
        ):
            raise ValueError("Incompatible radar sensor resolution")
        if scene.get("units") != "dB":
            raise ValueError("Baseline requires calibrated SAR dB input")
    return pairing_provenance(pre, post)


def load_local(path, event=None, cutoff=None):
    path = inference_path(path)
    manifest = read_config(path)
    if manifest.get("reference_labels") or manifest.get("purpose") == "evaluation":
        raise ValueError("Evaluation references cannot be used as inference inputs")
    manifest["event_date"] = event or manifest["event_date"]
    manifest["cutoff"] = cutoff or manifest.get("cutoff")
    manifest["pairing"] = validate_pair(
        manifest.get("pre"),
        manifest.get("post"),
        manifest["event_date"],
        manifest["cutoff"],
    )
    manifest["limitations"] = list(manifest.get("limitations") or [])
    for note in pairing_limitations(manifest["pairing"]):
        if note not in manifest["limitations"]:
            manifest["limitations"].append(note)
    for key in ("optical_pre", "optical_post"):
        if manifest.get(key):
            timestamp = utc(manifest[key]["timestamp"])
            if key == "optical_pre" and timestamp >= utc(manifest["event_date"]):
                raise ValueError("Optical background must precede event")
            if key == "optical_post" and timestamp <= utc(manifest["event_date"]):
                raise ValueError("Optical post observation must follow event")
            if manifest["cutoff"] and timestamp > utc(manifest["cutoff"]):
                raise ValueError("Optical observation exceeds cutoff")
    if manifest.get("polygon"):
        manifest["polygon"] = str(inference_path(path.parent / manifest["polygon"]))
    files = []
    for item in [
        *manifest["pre"],
        manifest["post"],
        *[
            manifest[k]
            for k in ("dem", "hand", "permanent_water", "optical_pre", "optical_post")
            if manifest.get(k)
        ],
    ]:
        p = inference_path(path.parent / item["path"])
        with rasterio.open(p) as src:
            if not src.crs:
                raise ValueError(f"Raster lacks CRS: {p}")
            item.update(
                path=str(p),
                crs=src.crs.to_string(),
                native_resolution=list(src.res),
                sha256=checksum(p),
            )
        files.append(item)
    # Additional same-event tracks for ascending/descending fusion (P04 T04.04).
    tracks = manifest.get("tracks") or []
    seen = {(manifest["post"]["relative_orbit"], manifest["post"]["pass"])}
    for track in tracks:
        key = (track["post"].get("relative_orbit"), track["post"].get("pass"))
        if key in seen:
            raise ValueError("Each fusion track must be a distinct relative orbit and pass")
        seen.add(key)
        track["pairing"] = validate_pair(
            track.get("pre"), track.get("post"), manifest["event_date"], manifest["cutoff"]
        )
        for scene in [*track["pre"], track["post"]]:
            p = inference_path(path.parent / scene["path"])
            with rasterio.open(p) as src:
                if not src.crs:
                    raise ValueError(f"Raster lacks CRS: {p}")
                scene.update(
                    path=str(p),
                    crs=src.crs.to_string(),
                    native_resolution=list(src.res),
                    sha256=checksum(p),
                )
            files.append(scene)
    for scene in [
        *manifest["pre"],
        manifest["post"],
        *[s for track in tracks for s in (*track["pre"], track["post"])],
    ]:
        masks = scene.get("masks")
        if not masks:
            continue
        p = inference_path(path.parent / masks["path"])
        with rasterio.open(p) as src:
            if not src.crs:
                raise ValueError(f"Raster lacks CRS: {p}")
            masks.update(path=str(p), crs=src.crs.to_string(), sha256=checksum(p))
        files.append(masks | {"source": f"{scene.get('source', 'SAR')} masks"})
    osm = manifest.get("osm")
    if not osm or not osm.get("snapshot"):
        raise DataUnavailable(
            "historical_map_missing",
            "Local vectors require a documented pre-event OSM snapshot",
        )
    osm["snapshot"] = check_osm_snapshot(osm["snapshot"], manifest["event_date"])
    for key in ("roads", "buildings", "settlements", "destinations"):
        if key in osm:
            p = inference_path(path.parent / osm[key])
            osm[key] = str(p)
            files.append(
                {
                    "path": str(p),
                    "sha256": checksum(p),
                    "source": "historical OSM",
                    "snapshot": osm["snapshot"],
                }
            )
    osm["cutoff_proof"] = edit_timestamp_proof(
        osm, osm["snapshot"], synthetic=manifest.get("synthetic") is True
    )
    manifest.update(
        processing_timestamp=now(),
        inputs=files,
        provider="local",
        software_versions={
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "rasterio",
                "geopandas",
                "shapely",
                "pyproj",
                "networkx",
            ]
        },
    )
    try:
        manifest["software_versions"]["earthengine-api"] = version("earthengine-api")
    except PackageNotFoundError:
        manifest["software_versions"]["earthengine-api"] = "not_installed"
    return manifest


def acquire_gee(geom, event, cutoff, config, cache, project):
    if not (project or os.environ.get("EE_PROJECT")):
        raise DataUnavailable(
            "credentials_required",
            "Set EE_PROJECT and authorize Earth Engine outside the pipeline",
        )
    try:
        import ee
    except ImportError as exc:
        raise DataUnavailable(
            "provider_dependency_unavailable",
            'Install the gee extra: pip install "himaroad[gee]"',
        ) from exc
    import requests

    from .gee import acquire

    try:
        return acquire(geom, event, cutoff, config, cache, project)
    except (ee.EEException, requests.RequestException) as exc:
        raise DataUnavailable("imagery_service_unavailable", str(exc)) from exc


def acquire_cdse(geom, event, cutoff, config, cache):
    if not (os.environ.get("SH_CLIENT_ID") and os.environ.get("SH_CLIENT_SECRET")):
        raise DataUnavailable(
            "credentials_required",
            "Set SH_CLIENT_ID and SH_CLIENT_SECRET (Sentinel Hub OAuth client) in .env",
        )
    import requests

    from .cdse import acquire

    try:
        return acquire(geom, event, cutoff, config, cache)
    except requests.RequestException as exc:
        raise DataUnavailable("imagery_service_unavailable", str(exc)) from exc
