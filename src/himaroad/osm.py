import json
import os
import time as clock
from collections import deque
from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import mapping

from .common import DataUnavailable, utc

FILTERS = {
    "roads": "highway=* and type:way",
    "buildings": "building=* and geometry:polygon",
    "settlements": "place in (village,hamlet,isolated_dwelling) and type:node",
    "destinations": (
        "(place in (town,city) or amenity in (hospital,clinic,doctors,health_post)"
        " or healthcare=*) and (type:node or geometry:polygon)"
    ),
}

# Per-feature time of the OSM version used: ohsome v2 extraction column
# `edit_timestamp` (docs.ohsome.org/ohsome-api/v2-rc/reference/data_model.html);
# ohsome v1 `properties=metadata` field `@lastEdit`.
EDIT_TIMESTAMP_FIELDS = ("edit_timestamp", "@lastEdit")

OHSOME_V2_URL = "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"
OHSOME_V1_URL = "https://api.ohsome.org/v1"
# ohsome v2-rc limits (docs.ohsome.org/ohsome-api/v2-rc/faq.html and
# reference/api.html): at most 15 requests in any sliding 60 s window (else 429
# "Rate Limit Exceeded"); an exhausted daily quota returns 403 "Quota Exceeded";
# extraction requests time out on the server after 240 s (504).
V2_REQUESTS_PER_WINDOW = 15
V2_WINDOW_S = 60.0
V2_READ_TIMEOUT_S = 250
V2_ATTEMPTS = 4
MAX_RETRY_WAIT_S = 300.0
_v2_sent = deque(maxlen=V2_REQUESTS_PER_WINDOW)
_sleep = clock.sleep


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def default_osm_snapshot(event_date: str) -> str:
    """00:00 UTC on the calendar day before the UTC date of the supplied event date."""
    day = utc(event_date).date() - timedelta(days=1)
    return _iso_z(datetime.combine(day, time(0), timezone.utc))


def check_osm_snapshot(snapshot: str, event_date: str) -> str:
    """Return the snapshot in ISO UTC; reject any snapshot later than the default."""
    limit = default_osm_snapshot(event_date)
    if utc(snapshot) > utc(limit):
        raise ValueError(
            f"OSM snapshot {snapshot} is later than {limit} (00:00 UTC on the day "
            f"before event date {event_date}); only earlier overrides are allowed"
        )
    return _iso_z(utc(snapshot))


def edit_timestamp_proof(
    osm_files: dict, snapshot: str, synthetic: bool = False
) -> dict:
    """Check that no feature in the OSM layer files was edited after the snapshot."""
    cutoff = utc(snapshot)
    latest = None
    checked = 0
    missing = 0
    fields = set()
    for kind in FILTERS:
        if not osm_files.get(kind):
            continue
        obj = json.loads(Path(osm_files[kind]).read_text())
        for feature in obj.get("features", []):
            properties = feature.get("properties") or {}
            checked += 1
            field = next((f for f in EDIT_TIMESTAMP_FIELDS if properties.get(f)), None)
            if field is None:
                missing += 1
                continue
            fields.add(field)
            edited = utc(properties[field])
            if edited > cutoff:
                feature_id = properties.get("@osmId", properties.get("id", "?"))
                # Fails closed as an honest incomplete run (review L8), never a crash.
                raise DataUnavailable(
                    "osm_cutoff_proof_failed",
                    f"OSM {kind} feature {feature_id} was edited at {_iso_z(edited)}, "
                    f"after the snapshot {_iso_z(cutoff)}",
                )
            latest = edited if latest is None or edited > latest else latest
    if missing and not synthetic:
        raise DataUnavailable(
            "osm_cutoff_proof_unavailable",
            f"{missing} of {checked} OSM features lack an edit timestamp",
        )
    return {
        "snapshot": _iso_z(cutoff),
        "max_edit_timestamp": _iso_z(latest) if latest else None,
        "features_checked": checked,
        "features_without_timestamp": missing,
        "passed": missing == 0,
        "timestamp_field": ",".join(sorted(fields)) or None,
    }


def _wait_for_v2_slot():
    if len(_v2_sent) == V2_REQUESTS_PER_WINDOW:
        wait = _v2_sent[0] + V2_WINDOW_S - clock.monotonic()
        if wait > 0:
            _sleep(wait)
    _v2_sent.append(clock.monotonic())


def _retry_wait(response, attempt):
    """Seconds to wait before a retry: Retry-After if sent, else exponential backoff."""
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                when = None
            if when is not None:
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    return min(60.0, 5.0 * 2**attempt)


def _error_text(response, key):
    try:
        body = response.json()
        text = str(body.get("error", "") if isinstance(body, dict) else body)[:200]
    except ValueError:
        text = response.text[:200]
    return text.replace(key, "***") if key else text


def _post_v2(payload, key):
    """POST one ohsome v2 extraction request with pacing and retries."""
    reason = "no attempt"
    for attempt in range(V2_ATTEMPTS):
        _wait_for_v2_slot()
        response = None
        try:
            response = requests.post(
                OHSOME_V2_URL,
                headers={"Authorization": key},
                json=payload,
                timeout=(15, V2_READ_TIMEOUT_S),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            reason = type(exc).__name__
        else:
            code = response.status_code
            if code == 200:
                return response
            error = _error_text(response, key)
            if code == 403 and "quota" in error.lower():
                reset = response.headers.get("x-ratelimit-reset")
                raise DataUnavailable(
                    "historical_map_quota_exceeded",
                    f"ohsome v2 daily quota exceeded (HTTP 403 {error!r}); "
                    f"quota resets at epoch {reset}",
                )
            if code in (401, 403):
                raise DataUnavailable(
                    "historical_map_credentials",
                    f"ohsome v2 HTTP {code} {error!r}: OHSOME_API_KEY is missing, "
                    "invalid or not authorised",
                )
            if code != 429 and code < 500:
                raise DataUnavailable(
                    "historical_map_service_unavailable",
                    f"ohsome v2 HTTP {code} {error!r}",
                )
            reason = f"HTTP {code} {error!r}"
        if attempt + 1 < V2_ATTEMPTS:
            wait = _retry_wait(response, attempt)
            if wait > MAX_RETRY_WAIT_S:
                raise DataUnavailable(
                    "historical_map_service_unavailable",
                    f"ohsome v2 {reason}; Retry-After {wait:.0f} s exceeds "
                    f"{MAX_RETRY_WAIT_S:.0f} s",
                )
            _sleep(wait)
    raise DataUnavailable(
        "historical_map_service_unavailable",
        f"ohsome v2 failed after {V2_ATTEMPTS} attempts: {reason}",
    )


def _v2_features(content):
    frame = gpd.read_parquet(
        BytesIO(content), to_pandas_kwargs={"maps_as_pydicts": "strict"}
    ).to_crs(4326)
    features = []
    # v2 names the geometry column `geom`; use the active geometry column.
    for geometry, (_, row) in zip(frame.geometry, frame.iterrows()):
        properties = dict(row["tags"])
        properties["id"] = f"{row['osm_type']}/{row['osm_id']}"
        edited = row.get("edit_timestamp")
        if not pd.isna(edited):
            properties["edit_timestamp"] = _iso_z(utc(edited.isoformat()))
        features.append(
            {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}
        )
    return {"type": "FeatureCollection", "features": features}


def ohsome_v1_end():
    """End of the ohsome v1 data (`toTimestamp` in /v1/metadata; frozen at
    2026-07-27T09:00Z when checked on 2026-10-13)."""
    try:
        response = requests.get(f"{OHSOME_V1_URL}/metadata", timeout=(15, 60))
        response.raise_for_status()
        return response.json()["extractRegion"]["temporalExtent"]["toTimestamp"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise DataUnavailable(
            "historical_map_unavailable",
            "OHSOME_API_KEY is not set and the ohsome v1 data end could not be "
            f"read ({type(exc).__name__}); no OSM snapshot can be proven",
        ) from exc


def _fetch_v1(buffered, snapshot, expression):
    payload = {
        "bpolys": json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": mapping(buffered)}
                ],
            }
        ),
        "time": snapshot,
        "filter": expression,
        "properties": "tags,metadata",
        "clipGeometry": "false",
    }
    response = None
    for attempt in range(3):
        try:
            response = requests.post(
                f"{OHSOME_V1_URL}/elements/geometry", data=payload, timeout=(15, 120)
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 2:
                raise DataUnavailable(
                    "historical_map_service_unavailable", str(exc)
                ) from exc
            continue
        if response.status_code < 500:
            break
    if response is None or response.status_code != 200:
        raise DataUnavailable(
            "historical_map_service_unavailable",
            f"ohsome v1 response {response.status_code if response else 'unavailable'}",
        )
    obj = response.json()
    if obj.get("type") != "FeatureCollection":
        raise DataUnavailable("historical_map_invalid", "Unexpected ohsome response")
    return obj


def buffered_aoi(geom, buffer_m):
    from .common import metric_crs

    frame = gpd.GeoSeries([geom], crs=4326)
    return frame.to_crs(metric_crs(geom)).buffer(buffer_m).to_crs(4326).iloc[0]


def fetch_historical(geom, snapshot, cache, buffer_m=5000):
    import hashlib

    buffered = buffered_aoi(geom, buffer_m)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = _iso_z(utc(snapshot))
    # "edit-ts-v1": cache files written before edit timestamps were kept are not reused.
    key = hashlib.sha256(
        ("edit-ts-v1" + json.dumps(FILTERS, sort_keys=True) + buffered.wkt + snapshot).encode()
    ).hexdigest()[:20]
    result = {
        "snapshot": snapshot,
        "source": "ohsome historical OSM",
        "network_buffer_m": buffer_m,
        "license": "OpenStreetMap contributors, ODbL 1.0",
    }
    paths = {kind: cache / f"osm-{key}-{kind}.geojson" for kind in FILTERS}
    api_key = os.environ.get("OHSOME_API_KEY")
    if not api_key and not all(path.exists() for path in paths.values()):
        # Never fall back to v1 for a snapshot its frozen data cannot represent.
        v1_end = ohsome_v1_end()
        if utc(snapshot) > utc(v1_end):
            raise DataUnavailable(
                "historical_map_unavailable",
                f"OHSOME_API_KEY is not set and the OSM snapshot {snapshot} is after "
                f"the end of the ohsome v1 data ({v1_end}); set OHSOME_API_KEY for "
                "ohsome v2 or choose an earlier snapshot",
            )
    for kind, expression in FILTERS.items():
        path = paths[kind]
        if not path.exists():
            if api_key:
                payload = {
                    "aoi": mapping(buffered),
                    "time": snapshot,
                    "filter": expression,
                    "clip": False,
                }
                response = _post_v2(payload, api_key)
                obj = _v2_features(response.content)
            else:
                obj = _fetch_v1(buffered, snapshot, expression)
            path.write_text(json.dumps(obj))
        result[kind] = str(path.resolve())
    result["cutoff_proof"] = edit_timestamp_proof(result, snapshot)
    return result


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_crosscheck(geom, snapshot, roads_path=None, buffer_m=5000):
    """Optional second source for the road snapshot: Overpass attic `[date:...]`.

    Light use only (public instance; wiki.openstreetmap.org/wiki/Overpass_API).
    Never a hard dependency: any failure is returned as status "unavailable".
    """
    snapshot = _iso_z(utc(snapshot))
    buffered = buffered_aoi(geom, buffer_m)
    polys = []
    for part in getattr(buffered, "geoms", [buffered]):
        coords = " ".join(f"{lat:.6f} {lon:.6f}" for lon, lat in part.exterior.coords)
        polys.append(f'way["highway"](poly:"{coords}");')
    query = (
        f'[out:json][timeout:180][date:"{snapshot}"];'
        f"({''.join(polys)});out meta;"
    )
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "HimaRoad educational prototype (cutoff cross-check)"},
            timeout=(15, 200),
        )
        if response.status_code != 200:
            reason = f"Overpass HTTP {response.status_code}"
            return {"status": "unavailable", "reason": reason}
        obj = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    if "error" in str(obj.get("remark", "")).lower():
        return {"status": "unavailable", "reason": f"Overpass remark: {obj['remark']}"}
    ways = [e for e in obj.get("elements", []) if e.get("type") == "way"]
    stamps = [utc(e["timestamp"]) for e in ways if e.get("timestamp")]
    ids = {f"way/{e['id']}" for e in ways}
    report = {
        "status": "ok",
        "source": "Overpass API attic (overpass-api.de)",
        "snapshot": snapshot,
        "ways": len(ways),
        "ways_without_timestamp": len(ways) - len(stamps),
        "ways_after_snapshot": sum(stamp > utc(snapshot) for stamp in stamps),
        "max_timestamp": _iso_z(max(stamps)) if stamps else None,
        "node_refs": sum(len(e.get("nodes", [])) for e in ways),
        "ways_missing_in_ohsome": None,
        "ways_missing_in_overpass": None,
    }
    if roads_path:
        features = json.loads(Path(roads_path).read_text()).get("features", [])
        ohsome_ids = {
            str((f.get("properties") or {}).get("id"))
            for f in features
            if str((f.get("properties") or {}).get("id", "")).startswith("way/")
        }
        report["ways_missing_in_ohsome"] = len(ids - ohsome_ids)
        report["ways_missing_in_overpass"] = len(ohsome_ids - ids)
    return report


def load_vectors(osm, crs):
    frames = {}
    for key in FILTERS:
        if not osm.get(key):
            frames[key] = gpd.GeoDataFrame({"id": []}, geometry=[], crs=crs)
            continue
        obj = json.loads(Path(osm[key]).read_text())
        if not obj.get("features"):
            frames[key] = gpd.GeoDataFrame({"id": []}, geometry=[], crs=crs)
            continue
        frame = gpd.GeoDataFrame.from_features(obj["features"], crs=4326)
        # Local GeoJSON is WGS84 by contract; avoid ambiguous legacy CRS members.
        frame["id"] = [
            str(row.get("@osmId", row.get("id", f"{key}/{i}")))
            for i, row in frame.iterrows()
        ]
        if frame["id"].duplicated().any():
            raise ValueError(f"Duplicate stable IDs in {key}")
        frames[key] = frame.to_crs(crs)
    return frames
