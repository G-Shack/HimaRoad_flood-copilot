import hashlib
import os
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import rasterio
import requests
from rasterio.merge import merge
from shapely.geometry import mapping

from .common import DataUnavailable, now, utc
from .acquire import pairing_limitations, pairing_provenance
from .cdse import event_limits
from .osm import buffered_aoi, check_osm_snapshot, default_osm_snapshot, fetch_historical

# Every GEE run carries this flag (A-D05, A-D14). GEE's COPERNICUS/S1_GRD scenes are
# terrain-corrected by Google, not with Copernicus GLO-30.
GEE_S1_LIMITATION = (
    "Sentinel-1 from Google Earth Engine COPERNICUS/S1_GRD, terrain-corrected by "
    "Google with SRTM: a flagged fallback, not the default CDSE gamma0 path; never "
    "used for the case study or judging (A-D14)."
)
# Copernicus GLO-30 on Earth Engine: an ImageCollection with band DEM, heights in
# metres above EGM2008 (EPSG:3855), 30 m. developers.google.com/earth-engine/
# datasets/catalog/COPERNICUS_DEM_GLO30
GLO30_COLLECTION = "COPERNICUS/DEM/GLO30"


def _download(image, geom, crs, scale, path):
    import ee

    if path.exists():
        with rasterio.open(path) as src:
            if src.crs and src.count and src.nodata == -9999:
                return
    # 512-pixel metric tiles stay well below EE's 32 MB / 10,000 dimension limits.
    import geopandas as gpd
    from shapely.geometry import box

    projected = gpd.GeoSeries([geom], crs=4326).to_crs(crs).iloc[0]
    xmin, ymin, xmax, ymax = projected.bounds
    step = scale * 512
    tiles = []
    import numpy as np

    for x in np.arange(xmin, xmax, step):
        for y in np.arange(ymin, ymax, step):
            tile = box(x, y, min(x + step, xmax), min(y + step, ymax))
            if not tile.intersects(projected):
                continue
            region = gpd.GeoSeries([tile], crs=crs).to_crs(4326).iloc[0]
            tile_path = path.with_name(f"{path.stem}-{len(tiles)}.tif")
            if not tile_path.exists():
                for attempt in range(3):
                    try:
                        url = image.unmask(-9999, sameFootprint=False).getDownloadURL(
                            {
                                "region": mapping(region),
                                "crs": crs.to_string(),
                                "scale": scale,
                                "format": "GEO_TIFF",
                            }
                        )
                        response = requests.get(url, timeout=(15, 120))
                        response.raise_for_status()
                        temp = tile_path.with_suffix(".partial")
                        temp.write_bytes(response.content)
                        with rasterio.open(temp, "r+") as src:
                            src.nodata = -9999
                            if not src.crs or not src.count:
                                raise ValueError("Invalid downloaded raster")
                        temp.replace(tile_path)
                        break
                    except (requests.RequestException, ee.EEException) as exc:
                        if attempt == 2:
                            raise DataUnavailable(
                                "imagery_download_failed", str(exc)
                            ) from exc
            tiles.append(tile_path)
    readers = [rasterio.open(p) for p in tiles]
    try:
        data, transform = merge(readers)
        profile = readers[0].profile | dict(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress="deflate",
            nodata=-9999,
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
    finally:
        for reader in readers:
            reader.close()


def acquire(geom, event, cutoff, config, cache, project):
    import ee

    from .common import metric_crs

    # Validate any OSM snapshot override before spending time on imagery.
    osm_snapshot = config.get("osm_snapshot")
    osm_snapshot = (
        check_osm_snapshot(osm_snapshot, event)
        if osm_snapshot
        else default_osm_snapshot(event)
    )
    discovery_started = perf_counter()
    project = project or os.environ.get("EE_PROJECT")
    if not project:
        raise DataUnavailable(
            "credentials_required",
            "Set EE_PROJECT and authorize Earth Engine outside the pipeline",
        )
    try:
        ee.Initialize(project=project)
        ee.data.setDeadline(60000)
        ee.data.setMaxRetries(3)
    except ee.EEException as exc:
        raise DataUnavailable("credentials_required", str(exc)) from exc
    start = utc(event)
    # Same event rule as the CDSE path: no event-day scene counts as post-event.
    pre_limit, post_start = event_limits(event, config.get("event_time"))
    end = min(
        utc(cutoff) if cutoff else utc(now()),
        start + timedelta(days=config.get("post_days", 20)),
    )
    if end <= start:
        raise DataUnavailable(
            "insufficient_observations", "Cutoff precedes post-event search window"
        )
    region = ee.Geometry(mapping(geom))
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    post_collection = collection.filterDate(start.isoformat(), end.isoformat()).sort(
        "system:time_start"
    )
    try:
        post_info = post_collection.limit(100).getInfo()["features"]
    except ee.EEException as exc:
        raise DataUnavailable("imagery_service_unavailable", str(exc)) from exc
    chosen = None
    for candidate in post_info:
        props = candidate["properties"]
        overlap = (
            ee.Image(candidate["id"])
            .geometry()
            .intersection(region, 100)
            .area(100)
            .divide(region.area(100))
            .getInfo()
        )
        if overlap < config.get("min_scene_coverage", 0.5):
            continue
        # Date only: the event may be any time that day, so post scenes follow the end
        # of the event date; an explicit event time allows same-day scenes after it.
        if props["system:time_start"] <= post_start.timestamp() * 1000:
            continue
        before = collection.filterDate(
            (start - timedelta(days=config.get("pre_days", 45))).isoformat(),
            pre_limit.isoformat(),
        )
        for field in (
            "relativeOrbitNumber_start",
            "orbitProperties_pass",
            "resolution_meters",
        ):
            before = before.filter(ee.Filter.eq(field, props[field]))
        features = (
            before.sort("system:time_start", False).limit(5).getInfo()["features"]
        )
        if features:
            chosen = candidate, features
            break
    if not chosen:
        raise DataUnavailable(
            "insufficient_observations", "No matched pre/post VV/VH IW orbit pair"
        )
    discovery_seconds = perf_counter() - discovery_started
    download_seconds = 0.0
    post_scene, pre_scenes = chosen
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    crs = metric_crs(geom)
    scale = max(20, config.get("resolution_m", 20))
    key = hashlib.sha256(
        ("nodata-v2" + geom.wkt + crs.to_string() + str(scale)).encode()
    ).hexdigest()[:16]

    def scene(item):
        nonlocal download_seconds
        p = item["properties"]
        path = cache / f"{key}-{item['id'].split('/')[-1]}.tif"
        image = ee.Image(item["id"]).select(["VV", "VH"])
        image = image.updateMask(image.select("VV").gt(-40))
        download_started = perf_counter()
        _download(image, geom, crs, scale, path)
        download_seconds += perf_counter() - download_started
        from datetime import datetime, timezone

        return dict(
            path=str(path),
            timestamp=datetime.fromtimestamp(
                p["system:time_start"] / 1000, timezone.utc
            ).isoformat(),
            scene_id=item["id"],
            source="COPERNICUS/S1_GRD",
            relative_orbit=p["relativeOrbitNumber_start"],
            **{"pass": p["orbitProperties_pass"]},
            instrument_mode=p["instrumentMode"],
            polarization="VV/VH",
            units="dB",
            native_sensor_resolution_m=p["resolution_meters"],
            platform_number=p.get("platform_number"),
            platform=(
                f"S1{p['platform_number']}" if p.get("platform_number") else "unknown"
            ),
        )

    manifest = dict(
        event_date=event,  # as supplied: a date stays a date (offset basis label)
        cutoff=end.isoformat(),
        bbox=list(geom.bounds),
        pre=[scene(f) for f in pre_scenes],
        post=scene(post_scene),
        synthetic=False,
        limitations=[
            "S1 GRD terrain corrected, not radiometrically terrain flattened.",
            GEE_S1_LIMITATION,
        ],
    )
    manifest["pairing"] = pairing_provenance(manifest["pre"], manifest["post"])
    manifest["limitations"] += pairing_limitations(manifest["pairing"])
    optional_started = perf_counter()
    # GLO-30 only (A-D06); HAND is derived from it and used only as a soft flag.
    dem_path = cache / f"{key}-glo30-dem.tif"
    dem_image = (
        ee.ImageCollection(GLO30_COLLECTION).filterBounds(region).select("DEM").mosaic()
    )
    try:
        # Buffered like the CDSE DEM, so HAND has upstream context.
        _download(dem_image, buffered_aoi(geom, config.get("network_buffer_m", 5000)), crs, 30, dem_path)
        manifest["dem"] = dict(
            path=str(dem_path),
            source=GLO30_COLLECTION,
            vertical_reference="EGM2008",
            native_resolution_m=30,
        )
    except (DataUnavailable, ee.EEException) as exc:
        manifest["limitations"].append(f"dem unavailable: {exc}")
    if "dem" in manifest:
        from .hand import derive_hand

        try:
            hand = derive_hand(dem_path, cache / f"{key}-glo30-hand.tif")
            manifest["hand"] = {**hand, "policy": "soft flag"}
        except ValueError as exc:
            manifest["limitations"].append(f"hand unavailable: {exc}")
    optical = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
    for key_name, begin, finish in [
        ("optical_pre", start - timedelta(days=config.get("pre_days", 45)), pre_limit),
        ("optical_post", post_start + timedelta(milliseconds=1), end),
    ]:
        try:
            choices = (
                optical.filterDate(begin.isoformat(), finish.isoformat())
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(1)
                .getInfo()["features"]
            )
            if choices:
                item = choices[0]
                path = cache / f"{key}-{item['id'].split('/')[-1]}-s2.tif"
                image = ee.Image(item["id"])
                scl = image.select("SCL")
                good = scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5)).Or(scl.eq(6))
                _download(
                    image.select(["B3", "B4", "B8", "B11", "SCL"])
                    .updateMask(good)
                    .toFloat(),
                    geom,
                    crs,
                    20,
                    path,
                )
                from datetime import datetime, timezone

                manifest[key_name] = dict(
                    path=str(path),
                    source="COPERNICUS/S2_SR_HARMONIZED",
                    timestamp=datetime.fromtimestamp(
                        item["properties"]["system:time_start"] / 1000, timezone.utc
                    ).isoformat(),
                    reflectance_scale=0.0001,
                )
            else:
                manifest["limitations"].append(f"{key_name} unavailable")
        except (DataUnavailable, ee.EEException) as exc:
            manifest["limitations"].append(f"{key_name} unavailable: {exc}")
    manifest["acquisition_timings_seconds"] = {
        "satellite_discovery": discovery_seconds,
        "sar_download": download_seconds,
        "optional_optical_terrain_acquisition": perf_counter() - optional_started,
    }
    manifest["osm"] = fetch_historical(
        geom,
        osm_snapshot,
        cache,
        config.get("network_buffer_m", 5000),
    )
    from .common import write_json

    path = cache / f"{key}-manifest.json"
    write_json(path, manifest)
    from .acquire import load_local

    result = load_local(path)
    result["provider"] = "gee"
    return result
