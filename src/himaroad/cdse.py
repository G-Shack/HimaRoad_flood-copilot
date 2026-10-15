"""CDSE provider (Phase 02, IMPLEMENTATION_PROMPT §6): STAC discovery, same-track
Sentinel-1 pairing, Sentinel Hub gamma0, Copernicus GLO-30 DEM and Sentinel-2 L2A.

Names below were checked against documentation.dataspace.copernicus.eu on 2026-10-13:
- token: APIs/SentinelHub/Overview/Authentication.html (client_credentials grant);
- STAC: APIs/STAC.html, /v1/search with CQL2-JSON; queryables checked live at
  /v1/collections/sentinel-1-grd/queryables;
- process endpoint and request layout: APIs/SentinelHub/Process/Examples/S1GRD.html;
- S1 dataFilter/processing/bands: APIs/SentinelHub/Data/S1GRD.html;
- DEM (type "dem", dataFilter.demInstance, egm): APIs/SentinelHub/Data/DEM.html;
- S2 (harmonizeValues, units DN): APIs/SentinelHub/Data/S2L2A.html;
- PU rules: APIs/SentinelHub/Overview/ProcessingUnit.html.
The PU response header name `x-processingunits-spent` is forum-sourced, not in the docs.
"""

import hashlib
import fcntl
import errno
import json
import math
import os
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
import requests
import shapely
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .acquire import pairing_limitations, pairing_provenance, scene_platform
from .common import DataUnavailable, checksum, data_root, metric_crs, now, utc, write_json
from .osm import check_osm_snapshot, default_osm_snapshot, fetch_historical

CODE_VERSION = "cdse-v1"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
# Sentinel Hub's own catalog: per-tile sensing times. A Sentinel-2 product's STAC
# datetime is its datatake start; the tile over an AOI is sensed minutes later.
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PU_HEADER = "x-processingunits-spent"
MAX_TILE_PX = 2500
MAX_CONNECTIONS = 4  # CDSE free tier: 4 concurrent connections
ATTEMPTS = 4
TOKEN_MARGIN_S = 60

S1_SOURCE = "CDSE Sentinel Hub S1GRD GAMMA0_TERRAIN (demInstance COPERNICUS_30)"
# Sentinel Hub's COPERNICUS_30 DEM layer is restricted to CCM-registered users (HTTP 403
# COMMON_INSUFFICIENT_PERMISSIONS for our account, 2026-10-13). The same GLO-30 DGED COG
# tiles are public (AWS Open Data copernicus-dem-30m; CDSE serves the same tile set at
# s3://eodata/auxdata/CopDEM_COG/copernicus-dem-30m/, which needs S3 keys).
DEM_SOURCE = "Copernicus DEM GLO-30 DGED COG (copernicus-dem-30m tiles)"
DEM_COG_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"
S2_SOURCE = "CDSE Sentinel Hub S2L2A"
# Only IW GRD high resolution (10 m pixel spacing); GRDM and other modes are rejected.
GRD_PRODUCTS = {"IW_GRDH_1S": 10}
S1_PROCESSING = {
    "backCoeff": "GAMMA0_TERRAIN",
    "orthorectify": True,
    "demInstance": "COPERNICUS_30",
}  # speckleFilter is not sent: the documented default is NONE.
S1_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask", "shadowMask", "localIncidenceAngle"]}],
    output: {id: "default", bands: 5, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask, s.shadowMask, s.localIncidenceAngle];
}
"""
S2_PROCESSING = {"harmonizeValues": "true"}
S2_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B04", "B08", "B11", "SCL", "dataMask", "B02"], units: "DN"}],
    output: {id: "default", bands: 7, sampleType: "UINT16"}
  };
}
function evaluatePixel(s) {
  return [s.B03, s.B04, s.B08, s.B11, s.SCL, s.dataMask, s.B02];
}
"""
# Same SCL encoding as flood.baseline: dark area, vegetation, not vegetated, water.
SCL_CLEAR = (2, 4, 5, 6)
S2_HARMONIZED = (
    "Sentinel Hub harmonizeValues=true with DN units: DN of processing baseline >= 04.00 "
    "are harmonised (BOA_ADD_OFFSET applied) so DN = 10000 x reflectance, negative values "
    "clamped to 0; converted to reflectance locally (DN x 0.0001)"
)

_sleep = time.sleep


def _wait(response, attempt):
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            return min(60.0, max(0.0, float(value)))
        except ValueError:
            pass
    return min(30.0, 2.0 * 2**attempt)


def _http(method, url, **kwargs):
    """One HTTP call with retries on 429/5xx and connection errors; raises otherwise."""
    kwargs.setdefault("timeout", (15, 180))
    response = None
    for attempt in range(ATTEMPTS):
        try:
            response = requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 == ATTEMPTS:
                raise
        else:
            if response.status_code != 429 and response.status_code < 500:
                return response
            if attempt + 1 == ATTEMPTS:
                break
        _sleep(_wait(response, attempt))
    if response.status_code == 429:
        raise DataUnavailable(
            "imagery_service_unavailable",
            f"CDSE rate limit or quota reached (HTTP 429 after {ATTEMPTS} attempts) "
            f"for {url}",
        )
    response.raise_for_status()
    return response


class SentinelHub:
    """Sentinel Hub OAuth client with PU log; the token is renewed 60 s before its
    `expires_in` (1800 s observed for the OAuth client on 2026-10-13)."""

    def __init__(self, budget_pu=500.0):
        self.budget_pu = float(budget_pu)
        self.token = None
        self.expires = 0.0
        self.lock = threading.Lock()
        self.log = {
            "processing_units_spent": 0.0,
            "processing_units_estimated": 0.0,
            "requests": 0,
            "requests_without_pu_header": 0,
            "cache_hits": 0,
            "budget_pu": self.budget_pu,
        }

    def _bearer(self):
        with self.lock:
            if self.token and time.monotonic() < self.expires - TOKEN_MARGIN_S:
                return self.token
            client_id = os.environ.get("SH_CLIENT_ID")
            secret = os.environ.get("SH_CLIENT_SECRET")
            if not (client_id and secret):
                raise DataUnavailable(
                    "credentials_required",
                    "Set SH_CLIENT_ID and SH_CLIENT_SECRET (Sentinel Hub OAuth client) in .env",
                )
            response = _http(
                "POST",
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": secret,
                },
            )
            if response.status_code in (400, 401, 403):
                raise DataUnavailable(
                    "credentials_required",
                    f"CDSE token request refused (HTTP {response.status_code})",
                )
            response.raise_for_status()
            body = response.json()
            self.token = body["access_token"]
            self.expires = time.monotonic() + float(body.get("expires_in", 600))
            return self.token

    def catalog_times(self, collection, bbox, begin, end):
        """{product id: tile sensing time} from the Sentinel Hub catalog (no PU cost)."""
        times, body = {}, {
            "collections": [collection],
            "bbox": list(bbox),
            "datetime": f"{_iso(begin)}/{_iso(end)}",
            "limit": 100,
        }
        while True:
            response = _http(
                "POST",
                CATALOG_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._bearer()}"},
                timeout=(15, 120),
            )
            if response.status_code != 200:
                raise DataUnavailable(
                    "imagery_service_unavailable",
                    f"Sentinel Hub catalog HTTP {response.status_code}",
                )
            page = response.json()
            for feature in page.get("features", []):
                # The catalog names products "<name>.SAFE"; STAC uses the bare name.
                times[feature["id"].removesuffix(".SAFE")] = feature["properties"]["datetime"]
            following = page.get("context", {}).get("next")
            if not following:
                return times
            body["next"] = following

    def process(self, request, estimate_pu):
        """POST one Process API request; returns the GeoTIFF bytes."""
        with self.lock:
            if self.log["processing_units_estimated"] + estimate_pu > self.budget_pu:
                # A configured cost limit, not a service outage (P07 drills).
                raise DataUnavailable(
                    "pu_budget_exceeded",
                    f"Sentinel Hub PU budget for this run ({self.budget_pu:g} PU, "
                    "config sentinel_hub_pu_budget) would be exceeded; "
                    f"{self.log['processing_units_estimated']:.1f} PU already estimated",
                )
            self.log["processing_units_estimated"] += estimate_pu
        response = _http(
            "POST",
            PROCESS_URL,
            json=request,
            headers={"Authorization": f"Bearer {self._bearer()}", "Accept": "image/tiff"},
        )
        if response.status_code in (401, 403):
            raise DataUnavailable(
                "credentials_required",
                f"Sentinel Hub refused the request (HTTP {response.status_code}): "
                f"{response.text[:300]}",
            )
        if response.status_code >= 400:
            raise DataUnavailable(
                "imagery_service_unavailable",
                f"Sentinel Hub HTTP {response.status_code}: {response.text[:300]}",
            )
        spent = response.headers.get(PU_HEADER)
        with self.lock:
            self.log["requests"] += 1
            if spent is None:
                self.log["requests_without_pu_header"] += 1
            else:
                self.log["processing_units_spent"] += float(spent)
        return response.content


# --- STAC ---------------------------------------------------------------------------


def _iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def stac_search(collection, geom, start, end, cql=None, limit=100):
    """All items of a CDSE STAC search, following `next` links."""
    body = {
        "collections": [collection],
        "intersects": mapping(geom),
        "datetime": f"{_iso(start)}/{_iso(end)}",
        "limit": limit,
    }
    if cql:
        body |= {"filter-lang": "cql2-json", "filter": cql}
    items, url, method = [], f"{STAC_URL}/search", "POST"
    for _ in range(100):
        response = _http(method, url, **({"json": body} if method == "POST" else {}))
        response.raise_for_status()
        page = response.json()
        items += page.get("features", [])
        link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not link:
            return items
        url, method = link["href"], link.get("method", "GET").upper()
        if method == "POST":
            body = (body | link["body"]) if link.get("merge") else link.get("body", body)
    raise DataUnavailable("imagery_service_unavailable", "STAC paging did not terminate")


S1_CQL = {
    "op": "and",
    "args": [
        {"op": "=", "args": [{"property": "sar:instrument_mode"}, "IW"]},
        {"op": "=", "args": [{"property": "product:type"}, "IW_GRDH_1S"]},
    ],
}


# --- pairing ------------------------------------------------------------------------


def _to_metric(geom, crs):
    to_crs = Transformer.from_crs(4326, crs, always_xy=True)
    return shapely.transform(geom, to_crs.transform, interleaved=False)


def _acquisitions(items, aoi):
    """Group STAC frames of one pass (platform + absolute orbit) and drop duplicates."""
    crs = metric_crs(aoi)
    aoi_m = _to_metric(aoi, crs)
    groups, rejected, seen = {}, [], {}
    for item in sorted(items, key=lambda i: (i["properties"]["datetime"], i["id"])):
        p = item["properties"]
        # Product name up to the datatake id (no CRC) identifies one acquisition frame.
        frame = "_".join(item["id"].split("_")[:8])
        if frame in seen:
            rejected.append(
                {"scene_id": item["id"], "reason": f"duplicate of {seen[frame]}"}
            )
            continue
        seen[frame] = item["id"]
        platform = scene_platform({"platform": p.get("platform")})
        key = (platform, p.get("sat:absolute_orbit"), p.get("sat:relative_orbit"))
        groups.setdefault(key, []).append(item)
    acquisitions = []
    for (platform, _, orbit), frames in groups.items():
        p = frames[0]["properties"]
        footprint = unary_union([shape(f["geometry"]) for f in frames])
        coverage = _to_metric(footprint, crs).intersection(aoi_m).area / aoi_m.area
        acquisitions.append(
            {
                "scene_id": "+".join(f["id"] for f in frames),
                "stac_item_ids": [f["id"] for f in frames],
                "timestamp": _iso(utc(p["datetime"])),
                "end_timestamp": _iso(
                    max(
                        utc(f["properties"].get("end_datetime") or f["properties"]["datetime"])
                        for f in frames
                    )
                ),
                "relative_orbit": orbit,
                "pass": str(p.get("sat:orbit_state", "")).upper(),
                "instrument_mode": p.get("sar:instrument_mode"),
                "polarizations": sorted(p.get("sar:polarizations") or []),
                "product_type": p.get("product:type"),
                "timeliness": p.get("product:timeliness"),
                "platform": platform,
                "platform_number": platform[-1] if platform != "unknown" else None,
                "aoi_coverage": round(coverage, 4),
            }
        )
    return sorted(acquisitions, key=lambda a: a["timestamp"]), rejected


def _track(a):
    return f"T{int(a['relative_orbit']):03d} {a['pass'].lower()}"


def event_limits(event: str, event_time: str | None = None):
    """(end of the pre-event period, start of the post-event period) for any sensor.

    With only an event date the event may be any time that day, so post-event
    observations start after the end of that date. An explicit event time splits
    them exactly; pre-event ones also precede the date boundary.
    """
    boundary = utc(event)
    pre_limit = min(boundary, utc(event_time)) if event_time else boundary
    post_start = utc(event_time) if event_time else boundary + timedelta(days=1)
    return pre_limit, post_start


def pair_s1(
    items: list,
    aoi,
    event: str,
    cutoff: str,
    pre_days: float,
    post_days: float,
    min_coverage: float,
    max_pre: int = 5,
    event_time: str | None = None,
    exclude_tracks: tuple = (),
) -> tuple:
    """First adequate post-event pass and its same-track pre-stack (most recent first).

    Returns (post or None, pre_stack, rejected) with a reason for every rejected item.
    Same track = relative orbit, pass, IW mode, VV+VH, GRDH (10 m) product.
    With only an event date, the event may happen any time that day, so a "post"
    scene must follow the end of the event date; an explicit event time lets
    same-day scenes after it count as post-event. Pre scenes precede both the
    date boundary and the event time.
    """
    boundary = utc(event)
    pre_limit, post_start = event_limits(event, event_time)
    end = min(utc(cutoff), boundary + timedelta(days=post_days))
    begin = boundary - timedelta(days=pre_days)
    acquisitions, rejected = _acquisitions(items, aoi)
    eligible = []
    for a in acquisitions:
        t = utc(a["timestamp"])
        reason = None
        if a["instrument_mode"] != "IW":
            reason = f"instrument mode {a['instrument_mode']} is not IW"
        elif a["product_type"] not in GRD_PRODUCTS:
            reason = f"product type {a['product_type']} is not IW GRDH"
        elif not {"VV", "VH"} <= set(a["polarizations"]):
            reason = f"polarisation {'+'.join(a['polarizations'])} lacks VV+VH"
        elif a["relative_orbit"] is None or a["pass"] not in ("ASCENDING", "DESCENDING"):
            reason = "relative orbit or orbit state missing"
        elif (a["relative_orbit"], a["pass"]) in exclude_tracks:
            reason = "track already used by another fused pair"
        elif t > utc(cutoff):
            reason = f"acquired after the cutoff {_iso(utc(cutoff))}"
        elif t > end:
            reason = f"after the post window ({post_days:g} days after the event date)"
        elif t < begin:
            reason = f"before the pre window ({pre_days:g} days)"
        elif pre_limit <= t <= post_start:
            reason = (
                "between the event-date boundary and the supplied event time"
                if event_time and t < post_start
                else "on the event date: it may precede the event (give an event time to use it)"
                if not event_time
                else "acquired exactly at the event time"
            )
        elif a["aoi_coverage"] < min_coverage:
            reason = f"AOI coverage {a['aoi_coverage']:.2f} < {min_coverage:g}"
        if reason:
            rejected.append({"scene_id": a["scene_id"], "reason": reason})
        else:
            eligible.append(a)
    pre_all = [a for a in eligible if utc(a["timestamp"]) < pre_limit]
    post_all = [a for a in eligible if utc(a["timestamp"]) > post_start]

    def same_track(a, b):
        return all(
            a[k] == b[k]
            for k in ("relative_orbit", "pass", "instrument_mode", "polarizations", "product_type")
        )

    post, stack = None, []
    for candidate in post_all:
        stack = sorted(
            (a for a in pre_all if same_track(a, candidate)),
            key=lambda a: a["timestamp"],
            reverse=True,
        )[:max_pre]
        if stack:
            post = candidate
            break
        rejected.append(
            {
                "scene_id": candidate["scene_id"],
                "reason": f"no same-track pre-event scene on {_track(candidate)}",
            }
        )
    chosen = {post["scene_id"]} if post else set()
    chosen |= {a["scene_id"] for a in stack}
    for a in eligible:
        if a["scene_id"] in chosen or any(r["scene_id"] == a["scene_id"] for r in rejected):
            continue
        if post is None:
            reason = "no adequate post-event scene on this track"
        elif utc(a["timestamp"]) > boundary:
            reason = f"later than the chosen post scene ({post['timestamp']})"
        elif not same_track(a, post):
            reason = f"different track from the chosen post ({_track(post)})"
        else:
            reason = f"older than the {max_pre} most recent same-track pre scenes"
        rejected.append({"scene_id": a["scene_id"], "reason": reason})
    return post, stack, rejected


def _windows(event, cutoff, config):
    boundary = utc(event)
    limit = utc(cutoff) if cutoff else utc(now())
    s1_end = min(limit, boundary + timedelta(days=config.get("post_days", 20)))
    s2_end = min(limit, boundary + timedelta(days=config.get("optical_post_days", 45)))
    if s1_end <= boundary:
        raise DataUnavailable(
            "insufficient_observations", "Cutoff precedes the post-event search window"
        )
    pre_start = boundary - timedelta(days=config.get("pre_days", 45))
    return boundary, limit, pre_start, s1_end, s2_end


def _summary(a, boundary):
    hours = (utc(a["timestamp"]) - boundary).total_seconds() / 3600
    return {
        "scene_id": a["scene_id"],
        "timestamp": a["timestamp"],
        "hours_after_event_date_boundary": round(hours, 2),
        "platform": a["platform"],
        "aoi_coverage": a["aoi_coverage"],
    }


def precheck(geom, event, cutoff, config) -> dict:
    """Sentinel-1/2 availability per track for an AOI and date; no imagery download."""
    started = perf_counter()
    boundary, limit, pre_start, s1_end, s2_end = _windows(event, cutoff, config)
    try:
        items = stac_search("sentinel-1-grd", geom, pre_start, s1_end, S1_CQL)
        s1_seconds = perf_counter() - started
        pre_limit, post_start = event_limits(event, config.get("event_time"))
        optical = {
            window: stac_search("sentinel-2-l2a", geom, begin, finish)
            for window, begin, finish in (
                ("pre", pre_start, pre_limit - timedelta(microseconds=1)),
                ("post", post_start + timedelta(microseconds=1), s2_end),
            )
        }
    except requests.RequestException as exc:
        raise DataUnavailable("imagery_service_unavailable", str(exc)) from exc
    args = dict(
        aoi=geom,
        event=event,
        cutoff=_iso(limit),
        pre_days=config.get("pre_days", 45),
        post_days=config.get("post_days", 20),
        min_coverage=config.get("min_scene_coverage", 0.5),
        max_pre=config.get("max_pre_scenes", 5),
        event_time=config.get("event_time"),
    )
    post, stack, rejected = pair_s1(items, **args)
    acquisitions, _ = _acquisitions(items, geom)
    tracks = []
    keys = {(a["relative_orbit"], a["pass"]) for a in acquisitions}
    for key in sorted(k for k in keys if k[0] is not None):
        own = [
            i
            for i in items
            if (i["properties"].get("sat:relative_orbit"),
                str(i["properties"].get("sat:orbit_state", "")).upper()) == key
        ]
        t_post, t_stack, _ = pair_s1(own, **args)
        mine = [a for a in acquisitions if (a["relative_orbit"], a["pass"]) == key]
        tracks.append(
            {
                "track": f"T{key[0]:03d}",
                "relative_orbit": key[0],
                "pass": key[1],
                "platforms": sorted({a["platform"] for a in mine}),
                "post_candidates": [
                    _summary(a, boundary) for a in mine if utc(a["timestamp"]) > post_start
                ],
                "pre_candidates": [
                    _summary(a, boundary) for a in mine if utc(a["timestamp"]) < pre_limit
                ],
                "pair": {
                    "post": t_post["scene_id"],
                    "pre": [a["scene_id"] for a in t_stack],
                }
                if t_post
                else None,
            }
        )
    clouds = {
        window: [
            {
                "scene_id": i["id"],
                "timestamp": i["properties"]["datetime"],
                "tile_cloud_cover": i["properties"].get("eo:cloud_cover"),
            }
            for i in sorted(found, key=lambda i: i["properties"].get("eo:cloud_cover", 101))[:5]
        ]
        for window, found in optical.items()
    }
    return {
        "event_date_boundary": _iso(boundary),
        "cutoff": _iso(limit),
        "search_windows": {
            "s1": [_iso(pre_start), _iso(s1_end)],
            "s2_post": [_iso(boundary), _iso(s2_end)],
        },
        "selected_pair": {
            "post": _summary(post, boundary),
            "pre": [_summary(a, boundary) for a in stack],
            "pairing": pairing_provenance(stack, post),
        }
        if post
        else None,
        "tracks": tracks,
        "rejected_scenes": rejected,
        "optical_least_cloudy": clouds,
        "note": "Hours are counted from the UTC event-date boundary, not the event time.",
        "timing_seconds": {
            "s1_discovery": round(s1_seconds, 3),
            "total": round(perf_counter() - started, 3),
        },
        "downloads": 0,
    }


# --- Sentinel Hub rasters -------------------------------------------------------------


def analysis_grid(geom, resolution, buffer_m=0.0):
    """UTM grid aligned with preprocess.prepare (same origin), optionally buffered."""
    crs = metric_crs(geom)
    left, bottom, right, top = _to_metric(geom, crs).bounds
    pad = math.ceil(buffer_m / resolution) * resolution
    left, top = left - pad, top + pad
    width = math.ceil((right + pad - left) / resolution)
    height = math.ceil((top - (bottom - pad)) / resolution)
    return {
        "crs": crs,
        "transform": from_origin(left, top, resolution, resolution),
        "width": width,
        "height": height,
        "resolution": resolution,
    }


def tiles(width, height, max_px=MAX_TILE_PX):
    """(col, row, w, h) windows covering the grid, none wider or taller than max_px."""
    nx, ny = math.ceil(width / max_px), math.ceil(height / max_px)
    xs = [round(i * width / nx) for i in range(nx + 1)]
    ys = [round(j * height / ny) for j in range(ny + 1)]
    return [
        (xs[i], ys[j], xs[i + 1] - xs[i], ys[j + 1] - ys[j])
        for j in range(ny)
        for i in range(nx)
    ]


def estimate_pu(width, height, input_bands, float32=False, factor=1.0):
    """PU estimate from the documented multiplication factors (dataMask not counted)."""
    area = max(0.01, width * height / (512 * 512))
    return max(0.005, area * input_bands / 3 * (2 if float32 else 1) * factor)


def _grid_key(grid):
    return {
        "crs": grid["crs"].to_string(),
        "transform": list(grid["transform"])[:6],
        "width": grid["width"],
        "height": grid["height"],
    }


def cache_key(parts: dict) -> str:
    identity = json.dumps(parts | {"code_version": CODE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


# Only allowed inputs (IMPLEMENTATION_PROMPT §0.2). The CDSE account now has CCM access
# (A-D24): its commercial very-high-resolution imagery is forbidden and must never be requested.
ALLOWED_PROCESS_TYPES = frozenset({"sentinel-1-grd", "sentinel-2-l2a"})


def fetch(client, data, evalscript, grid, bands, dtype, estimate):
    """Request the grid tile by tile (<= MAX_TILE_PX) and mosaic into (bands, h, w)."""
    if data.get("type") not in ALLOWED_PROCESS_TYPES:
        raise ValueError(f"Sentinel Hub collection {data.get('type')!r} is not an allowed input")
    out = np.zeros((bands, grid["height"], grid["width"]), dtype=dtype)
    res = grid["resolution"]
    left, top = grid["transform"].c, grid["transform"].f
    epsg = grid["crs"].to_epsg()
    for col, row, w, h in tiles(grid["width"], grid["height"]):
        x0, y1 = left + col * res, top - row * res
        request = {
            "input": {
                "bounds": {
                    "bbox": [x0, y1 - h * res, x0 + w * res, y1],
                    "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
                },
                "data": [data],
            },
            "output": {
                "width": w,
                "height": h,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }
        content = client.process(request, estimate(w, h))
        with MemoryFile(content) as memory, memory.open() as src:
            if (src.count, src.height, src.width) != (bands, h, w):
                raise DataUnavailable(
                    "imagery_service_unavailable",
                    f"Sentinel Hub returned {src.count}x{src.height}x{src.width}, "
                    f"expected {bands}x{h}x{w}",
                )
            out[:, row : row + h, col : col + w] = src.read()
    return out


def _write(path, data, grid, descriptions):
    profile = dict(
        driver="GTiff",
        width=grid["width"],
        height=grid["height"],
        count=data.shape[0],
        dtype="float32",
        crs=grid["crs"],
        transform=grid["transform"],
        nodata=np.nan,
        compress="deflate",
    )
    partial = path.with_suffix(".partial.tif")
    with rasterio.open(partial, "w", **profile) as dst:
        dst.write(data.astype("float32"))
        for index, name in enumerate(descriptions, start=1):
            dst.set_band_description(index, name)
    partial.replace(path)


def to_db(linear, data_mask):
    """10*log10 of linear power; NaN where dataMask == 0 or the value is <= 0."""
    linear = np.asarray(linear, dtype="float64")
    valid = (np.asarray(data_mask) == 1) & np.isfinite(linear) & (linear > 0)
    db = np.full(linear.shape, np.nan, dtype="float32")
    db[valid] = 10 * np.log10(linear[valid])
    return db


def _aoi_mask(geom, grid):
    return geometry_mask(
        [_to_metric(geom, grid["crs"])],
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        invert=True,
    )


def s1_scene(client, acquisition, grid, geom, folder):
    """Gamma0 VV/VH dB + masks GeoTIFFs for one acquisition (cached)."""
    start = utc(acquisition["timestamp"]) - timedelta(seconds=60)
    end = utc(acquisition["end_timestamp"]) + timedelta(seconds=60)
    data = {
        "type": "sentinel-1-grd",
        "dataFilter": {
            "timeRange": {"from": _iso(start), "to": _iso(end)},
            "acquisitionMode": "IW",
            "polarization": "DV",
            "resolution": "HIGH",
            "orbitDirection": acquisition["pass"],
        },
        "processing": S1_PROCESSING,
    }
    key = cache_key(
        {"items": acquisition["stac_item_ids"], "data": data, "evalscript": S1_EVALSCRIPT}
        | _grid_key(grid)
    )
    path, masks = folder / f"{key}-s1.tif", folder / f"{key}-s1-masks.tif"
    if path.exists() and masks.exists():
        with client.lock:
            client.log["cache_hits"] += 1
    else:
        raw = fetch(
            client,
            data,
            S1_EVALSCRIPT,
            grid,
            5,
            "float32",
            lambda w, h: estimate_pu(w, h, 4, float32=True, factor=2.5),
        )
        mask = raw[2]
        db = np.stack([to_db(raw[0], mask), to_db(raw[1], mask)])
        inside = _aoi_mask(geom, grid)
        if not np.isfinite(db[0][inside]).any():
            raise DataUnavailable(
                "imagery_service_unavailable",
                "Sentinel Hub returned no valid pixels over the AOI for "
                f"{acquisition['scene_id']}",
            )
        aux = np.stack(
            [mask, np.where(mask == 1, raw[3], np.nan), np.where(mask == 1, raw[4], np.nan)]
        )
        _write(masks, aux, grid, ["dataMask", "shadowMask", "localIncidenceAngle"])
        _write(path, db, grid, ["VV_dB", "VH_dB"])
    with rasterio.open(path) as src:
        valid = float(np.isfinite(src.read(1)[_aoi_mask(geom, grid)]).mean())
    return {
        "path": str(path),
        "timestamp": acquisition["timestamp"],
        "scene_id": acquisition["scene_id"],
        "stac_item_ids": acquisition["stac_item_ids"],
        "source": S1_SOURCE,
        "relative_orbit": int(acquisition["relative_orbit"]),
        "pass": acquisition["pass"],
        "instrument_mode": "IW",
        "polarization": "VV/VH",
        "units": "dB",
        "native_sensor_resolution_m": GRD_PRODUCTS[acquisition["product_type"]],
        "platform": acquisition["platform"],
        "platform_number": acquisition["platform_number"],
        "timeliness": acquisition["timeliness"],
        "aoi_coverage": acquisition["aoi_coverage"],
        "aoi_valid_fraction": round(valid, 4),
        "processing": S1_PROCESSING
        | {"speckleFilter": "NONE (default)", "timeRange": data["dataFilter"]["timeRange"]},
        "masks": {
            "path": str(masks),
            "bands": ["dataMask", "shadowMask", "localIncidenceAngle"],
        },
    }


def glo30_tile_names(bounds):
    """1-degree GLO-30 tile names covering lon/lat bounds (west, south, east, north)."""
    west, south, east, north = bounds
    names = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            names.append(f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM")
    return names


def _cached_dem_tile(name, cache, limit):
    cache = Path(cache).expanduser().resolve()
    if not cache.is_relative_to(data_root() / "cache"):
        raise ValueError("GLO-30 cache must be inside HIMAROAD_DATA_DIR/cache")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("GLO-30 cache limit must be a positive byte count")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (name + ".tif")
    sidecar = cache / (name + ".sha256.json")
    try:
        descriptor = os.open(cache / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("GLO-30 cache contains a symlink") from exc
        raise
    with os.fdopen(descriptor, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = list(cache.rglob("*"))
        if any(p.is_symlink() for p in entries):
            raise ValueError("GLO-30 cache contains a symlink")
        retained = sum(p.stat().st_size for p in entries if p.is_file())
        if retained > limit:
            raise DataUnavailable("imagery_service_unavailable", "GLO-30 cache storage limit exceeded")
        if path.exists() or sidecar.exists():
            try:
                record = json.loads(sidecar.read_text())
                valid = (record["size_bytes"] == path.stat().st_size
                         and record["sha256"] == checksum(path))
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise DataUnavailable("imagery_service_unavailable", f"Incomplete or corrupt GLO-30 cache tile {name}")
            return path, record, retained
        response = _http("GET", DEM_COG_URL.format(name=name), stream=True)
        temporary = None
        try:
            if response.status_code == 404:
                return None, None, retained
            if response.status_code != 200:
                raise DataUnavailable("imagery_service_unavailable", f"GLO-30 tile {name}: HTTP {response.status_code}")
            advertised = response.headers.get("Content-Length")
            advertised = int(advertised) if advertised is not None else None
            # Reserve the exact sidecar size before streaming; the digest is always 64 hex characters.
            reserve = len(json.dumps({"size_bytes": advertised or limit, "sha256": "0" * 64}).encode())
            if advertised is not None and (advertised <= 0 or retained + advertised + reserve > limit):
                raise DataUnavailable("imagery_service_unavailable", "GLO-30 cache storage limit exceeded or invalid Content-Length")
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".download-", delete=False) as out:
                temporary = Path(out.name)
                size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if retained + size + reserve > limit:
                        raise DataUnavailable("imagery_service_unavailable", "GLO-30 cache storage limit exceeded")
                    out.write(chunk)
            if size <= 0 or (advertised is not None and size != advertised):
                raise DataUnavailable("imagery_service_unavailable", f"Incomplete GLO-30 download {name}")
            try:
                with rasterio.open(temporary) as source:
                    if source.count != 1 or source.crs is None:
                        raise DataUnavailable("imagery_service_unavailable", f"Invalid GLO-30 tile {name}")
                    # Opening the TIFF alone does not detect damaged compressed blocks.
                    for _, window in source.block_windows(1):
                        source.read(1, window=window)
            except rasterio.errors.RasterioError as exc:
                raise DataUnavailable("imagery_service_unavailable", f"Invalid GLO-30 tile {name}: {exc}") from exc
            record = {"size_bytes": size, "sha256": checksum(temporary)}
            payload = json.dumps(record).encode()
            temporary.replace(path)
            temporary = None
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".digest-", delete=False) as out:
                temporary = Path(out.name)
                out.write(payload)
            temporary.replace(sidecar)
            temporary = None
            return path, record, retained + size + len(payload)
        finally:
            response.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def glo30_dem(grid, cache=None, cache_limit_bytes=512_000_000):
    """GLO-30 heights (m, EGM2008) bilinearly resampled onto the grid; tile record."""
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    out = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    bounds = array_bounds(grid["height"], grid["width"], grid["transform"])
    used, missing, records = [], [], {}
    retained = 0
    for name in glo30_tile_names(transform_bounds(grid["crs"], 4326, *bounds)):
        url = DEM_COG_URL.format(name=name)
        if cache is not None:
            source, record, retained = _cached_dem_tile(name, cache, cache_limit_bytes)
            if source is None:
                missing.append(name)
                continue
            records[name] = record
        else:
            head = _http("HEAD", url, timeout=(15, 60))
            if head.status_code == 404:
                missing.append(name)  # GLO-30 has no tiles over open ocean.
                continue
            if head.status_code != 200:
                raise DataUnavailable("imagery_service_unavailable", f"GLO-30 tile {name}: HTTP {head.status_code}")
            source = "/vsicurl/" + url
        tile = np.full_like(out, np.nan)
        try:
            src = rasterio.open(source)
        except rasterio.errors.RasterioIOError as exc:
            raise DataUnavailable("imagery_service_unavailable", f"GLO-30 tile {name}: {exc}") from exc
        with src:
            reproject(
                rasterio.band(src, 1),
                tile,
                src_nodata=src.nodata,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        out = np.where(np.isnan(out), tile, out)
        used.append(name)
    if not used:
        raise DataUnavailable("insufficient_observations", "No Copernicus GLO-30 tile covers the AOI")
    provenance = {"tiles": used, "missing_tiles": missing, "url_template": DEM_COG_URL}
    if cache is not None:
        provenance.update(tile_records=records, cache_retained_bytes=retained)
    return out, provenance


def dem_and_hand(client, geom, grid, folder):
    """GLO-30 DEM (orthometric, EGM2008) on the buffered grid and HAND as a soft flag."""
    from .hand import derive_hand

    from .hand import HAND_VERSION, STREAM_AREA_KM2

    key = cache_key(
        {"dem": "glo30-cog-v1", "url": DEM_COG_URL, "hand": HAND_VERSION, "stream_km2": STREAM_AREA_KM2}
        | _grid_key(grid)
    )
    path, hand_path = folder / f"{key}-dem.tif", folder / f"{key}-hand.tif"
    record_path, hand_record = folder / f"{key}-dem.json", folder / f"{key}-hand.json"
    if path.exists() and record_path.exists():
        with client.lock:
            client.log["cache_hits"] += 1
        tiles_used = json.loads(record_path.read_text())
    else:
        heights, tiles_used = glo30_dem(grid)
        _write(path, heights[None], grid, ["elevation_m"])
        write_json(record_path, tiles_used)
    if hand_path.exists() and hand_record.exists():
        hand = json.loads(hand_record.read_text())
    else:
        hand = derive_hand(path, hand_path)
        write_json(hand_record, hand)
    dem = {
        "path": str(path),
        "source": DEM_SOURCE,
        "vertical_reference": "EGM2008",
        "native_resolution_m": 30,
        "grid_resolution_m": grid["resolution"],
        "resampling": "BILINEAR",
        **tiles_used,
    }
    return dem, hand | {"path": str(hand_path), "policy": "soft flag"}


def s2_search(geom, begin, end):
    """S2 L2A acquisitions (tiles of one platform pass merged), least cloudy first."""
    groups = {}
    for item in stac_search("sentinel-2-l2a", geom, begin, end):
        p = item["properties"]
        groups.setdefault((p.get("platform"), p["datetime"]), []).append(item)
    crs = metric_crs(geom)
    aoi = _to_metric(geom, crs)
    found = [
        {
            "scene_id": "+".join(i["id"] for i in items),
            "item_ids": [i["id"] for i in items],
            "aoi_coverage": round(
                unary_union([_to_metric(shape(i["geometry"]), crs) for i in items])
                .intersection(aoi)
                .area
                / aoi.area,
                4,
            ),
            "timestamp": _iso(utc(dt)),
            "platform": platform,
            "tile_cloud_cover": max(i["properties"].get("eo:cloud_cover", 100) for i in items),
            "processing_baseline": items[0]["properties"].get("processing:version"),
        }
        for (platform, dt), items in groups.items()
    ]
    return sorted(found, key=lambda c: (c["tile_cloud_cover"], c["timestamp"]))


def s2_scene(client, candidate, grid, geom, folder):
    """Reflectance (green, red, nir, swir, blue) + SCL GeoTIFF and the AOI clear fraction."""
    # +-60 s around the tile sensing times: other passes the same day are >= 10 min away.
    sensed = sorted(utc(value) for value in candidate["tile_times"])
    data = {
        "type": "sentinel-2-l2a",
        "dataFilter": {
            "timeRange": {
                "from": _iso(sensed[0] - timedelta(seconds=60)),
                "to": _iso(sensed[-1] + timedelta(seconds=60)),
            }
        },
        "processing": S2_PROCESSING,
    }
    key = cache_key(
        {"items": candidate["scene_id"], "data": data, "evalscript": S2_EVALSCRIPT}
        | _grid_key(grid)
    )
    path = folder / f"{key}-s2.tif"
    if path.exists():
        with client.lock:
            client.log["cache_hits"] += 1
    else:
        raw = fetch(
            client,
            data,
            S2_EVALSCRIPT,
            grid,
            7,
            "uint16",
            lambda w, h: estimate_pu(w, h, 6),
        ).astype("float32")
        valid = raw[5] == 1
        bands = np.where(valid, raw[:4] * 0.0001, np.nan)
        scl = np.where(valid, raw[4], np.nan)
        blue = np.where(valid, raw[6] * 0.0001, np.nan)  # B02 for the bare-soil index (P04)
        names = ["B03", "B04", "B08", "B11", "SCL", "B02"]
        _write(path, np.concatenate([bands, scl[None], blue[None]]), grid, names)
    with rasterio.open(path) as src:
        scl = src.read(5)
    inside = _aoi_mask(geom, grid)
    clear = float(np.isin(scl[inside], SCL_CLEAR).mean()) if inside.any() else 0.0
    return {
        "path": str(path),
        "source": S2_SOURCE,
        "scene_id": candidate["scene_id"],
        "timestamp": _iso(sensed[0]),  # tile sensing time (Sentinel Hub catalog)
        "datatake_start": candidate["timestamp"],
        "aoi_coverage": candidate["aoi_coverage"],
        "platform": candidate["platform"],
        "bands": {"green": 1, "red": 2, "nir": 3, "swir": 4, "quality": 5, "blue": 6},
        "reflectance_scale": 1.0,
        "clear_fraction": round(clear, 4),
        "cloud_mask": "SCL",
        "clear_scl_classes": list(SCL_CLEAR),
        "harmonized_offset": S2_HARMONIZED,
        "processing_baseline": candidate["processing_baseline"],
        "tile_cloud_cover": candidate["tile_cloud_cover"],
    }


def _optical(client, geom, grid, folder, windows, config, s1_post, limitations):
    per_window = config.get("optical_candidates", 3)
    threshold = config.get("min_optical_clear_fraction", 0.3)
    chosen, record = {}, {}
    coverage = config.get("min_scene_coverage", 0.5)
    settle = timedelta(days=config.get("optical_search_cache_settle_days", 3))
    for window, begin, end in windows:
        # Warm reruns reuse the STAC search of a settled window; a live window is always searched.
        search = folder / (
            cache_key({"search": "sentinel-2-l2a", "aoi": geom.wkt, "window": [_iso(begin), _iso(end)]})
            + "-search.json"
        )
        if search.exists():
            cached = json.loads(search.read_text())
            found = cached["candidates"]
            limitations.append(
                f"Sentinel-2 {window}-window catalog search reused from the cache (searched "
                f"{cached['searched_at_utc']}); scenes published later are not considered."
            )
        else:
            searched_at = utc(now())
            found = s2_search(geom, begin, end)
            if end <= searched_at - settle:
                write_json(search, {"searched_at_utc": _iso(searched_at), "candidates": found})
        times = {}
        if found:
            catalog = folder / (
                cache_key({"catalog": "sentinel-2-l2a", "bbox": geom.bounds, "window": [_iso(begin), _iso(end)]})
                + "-catalog.json"
            )
            if catalog.exists():
                times = {k.removesuffix(".SAFE"): v for k, v in json.loads(catalog.read_text()).items()}
            else:
                times = client.catalog_times("sentinel-2-l2a", geom.bounds, begin, end)
                write_json(catalog, times)
        usable, rejected = [], []
        for candidate in found:
            if candidate["aoi_coverage"] < coverage:
                rejected.append({"scene_id": candidate["scene_id"], "reason": f"AOI coverage {candidate['aoi_coverage']:.2f} < {coverage:g}"})
            elif any(item not in times for item in candidate["item_ids"]):
                rejected.append({"scene_id": candidate["scene_id"], "reason": "tile sensing time outside the window, or not in the Sentinel Hub catalog"})
            else:
                usable.append(candidate | {"tile_times": [times[i] for i in candidate["item_ids"]]})
        scenes = [s2_scene(client, c, grid, geom, folder) for c in usable[:per_window]]
        record[window] = {
            "searched": len(found),
            "rejected": rejected,
            "fetched": [
                {
                    k: s[k]
                    for k in ("scene_id", "timestamp", "aoi_coverage", "tile_cloud_cover", "clear_fraction")
                }
                for s in scenes
            ],
        }
        if not scenes:
            limitations.append(
                f"optical_{window} unavailable: no Sentinel-2 L2A scene in the window"
                if not found
                else f"optical_{window} unavailable: all {len(found)} Sentinel-2 L2A candidates "
                "were rejected (reasons in manifest optical_candidates)"
            )
            continue
        if window == "pre":
            pick = max(scenes, key=lambda s: (s["clear_fraction"], s["timestamp"]))
            if pick["clear_fraction"] < threshold:
                limitations.append(
                    f"The clearest pre-event Sentinel-2 candidate ({pick['timestamp']}) has an "
                    f"AOI clear fraction of {pick['clear_fraction']:.2f} < {threshold:g}."
                )
        else:
            good = sorted(
                (s for s in scenes if s["clear_fraction"] >= threshold),
                key=lambda s: s["timestamp"],
            )
            pick = good[0] if good else max(scenes, key=lambda s: s["clear_fraction"])
            if not good:
                limitations.append(
                    "No post-event Sentinel-2 scene reached an AOI clear fraction of "
                    f"{threshold:g} among the {len(scenes)} least cloudy candidates; the clearest "
                    f"({pick['timestamp']}, clear fraction {pick['clear_fraction']:.2f}) is used."
                )
        chosen[f"optical_{window}"] = pick
        gap = (utc(pick["timestamp"]) - utc(s1_post["timestamp"])).total_seconds() / 86400
        pick["days_from_s1_post"] = round(gap, 2)
        limitations.append(
            f"Sentinel-2 {window}-event scene {pick['timestamp']} is {abs(gap):.1f} days "
            f"{'after' if gap >= 0 else 'before'} the Sentinel-1 post scene {s1_post['timestamp']}."
        )
    return chosen, record


def acquire(geom, event, cutoff, config, cache):
    """CDSE acquisition; writes a manifest to the cache and returns load_local(manifest)."""
    from .acquire import load_local

    osm_snapshot = config.get("osm_snapshot")
    osm_snapshot = (
        check_osm_snapshot(osm_snapshot, event) if osm_snapshot else default_osm_snapshot(event)
    )
    started = perf_counter()
    boundary, limit, pre_start, s1_end, s2_end = _windows(event, cutoff, config)
    items = stac_search("sentinel-1-grd", geom, pre_start, s1_end, S1_CQL)
    post, stack, rejected = pair_s1(
        items,
        geom,
        event,
        _iso(limit),
        config.get("pre_days", 45),
        config.get("post_days", 20),
        config.get("min_scene_coverage", 0.5),
        config.get("max_pre_scenes", 5),
        config.get("event_time"),
    )
    if post is None:
        reasons = "; ".join(sorted({r["reason"] for r in rejected})) or "no scenes found"
        raise DataUnavailable(
            "insufficient_observations",
            f"No same-track IW GRDH VV+VH pre/post pair: {reasons}",
        )
    discovery = perf_counter() - started
    resolution = float(config.get("resolution_m", 20))
    if resolution < 10:
        raise ValueError("CDSE Sentinel-1 grid resolution must be >= 10 m")
    folder = Path(cache) / "cdse"
    folder.mkdir(parents=True, exist_ok=True)
    grid = analysis_grid(geom, resolution)
    client = SentinelHub(config.get("sentinel_hub_pu_budget", 500))
    t = perf_counter()
    with ThreadPoolExecutor(max_workers=min(MAX_CONNECTIONS, 1 + len(stack))) as pool:
        scenes = list(
            pool.map(lambda a: s1_scene(client, a, grid, geom, folder), [*stack, post])
        )
    # Ascending/descending fusion (P04 T04.04): the next adequate pairs on other tracks.
    fused, track_notes = [], []
    used = {(post["relative_orbit"], post["pass"])}
    for _ in range(int(config.get("fusion_max_tracks", 2))):
        extra_post, extra_stack, _ = pair_s1(
            items,
            geom,
            event,
            _iso(limit),
            config.get("pre_days", 45),
            config.get("post_days", 20),
            config.get("min_scene_coverage", 0.5),
            config.get("max_pre_scenes", 5),
            config.get("event_time"),
            exclude_tracks=tuple(used),
        )
        if extra_post is None:
            break
        gap = (utc(extra_post["timestamp"]) - utc(post["timestamp"])).total_seconds() / 86400
        used.add((extra_post["relative_orbit"], extra_post["pass"]))
        if gap > config.get("fusion_max_post_gap_days", 12):
            track_notes.append(
                f"Track {_track(extra_post)} not fused: its first post scene is {gap:.1f} days after "
                f"the primary post scene (> {config.get('fusion_max_post_gap_days', 12)})."
            )
            continue
        fused.append((extra_post, extra_stack))
    with ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as pool:
        tracks = [
            {
                "pre": list(pool.map(lambda a: s1_scene(client, a, grid, geom, folder), stack_)),
                "post": s1_scene(client, post_, grid, geom, folder),
            }
            for post_, stack_ in fused
        ]
    sar_seconds = perf_counter() - t
    t = perf_counter()
    # DEM buffer: the OSM network buffer, so HAND has upstream context and the DEM covers
    # the routed network extent; snapped to whole pixels so grids stay aligned.
    dem_grid = analysis_grid(geom, resolution, config.get("network_buffer_m", 5000))
    dem, hand = dem_and_hand(client, geom, dem_grid, folder)
    terrain_seconds = perf_counter() - t
    limitations = [
        "Sentinel-1 gamma0 terrain-flattened backscatter processed on request from raw GRD "
        "via CDSE Sentinel Hub (COPERNICUS_30).",
        "Baseline dB thresholds were defined for sigma0-style GRD, not re-derived for gamma0 "
        "(Phase 04).",
        "Sentinel Hub provides no layover mask; shadowMask and localIncidenceAngle are stored "
        "per scene but not yet applied (Phase 04).",
        "Copernicus GLO-30 is a surface model (DSM) on EGM2008; slope and HAND inherit its "
        "canopy and building heights. dem.tif comes from the GLO-30 DGED COG tiles; Sentinel "
        "Hub's terrain correction uses its COPERNICUS_30 DEM (GLO-90 infill where GLO-30 is "
        "not released).",
        "Sentinel-2 cloud mask is the Sen2Cor SCL classification (classes 2, 4, 5, 6 clear); "
        "OmniCloudMask is not applied.",
    ]
    t = perf_counter()
    # Same event rule as Sentinel-1: no event-day optical scene counts as post-event.
    pre_limit, post_start = event_limits(event, config.get("event_time"))
    optical, candidates = _optical(
        client,
        geom,
        grid,
        folder,
        [
            ("pre", pre_start, pre_limit - timedelta(microseconds=1)),
            ("post", post_start + timedelta(microseconds=1), s2_end),
        ],
        config,
        scenes[-1],
        limitations,
    )
    optical_seconds = perf_counter() - t
    t = perf_counter()
    osm = fetch_historical(geom, osm_snapshot, cache, config.get("network_buffer_m", 5000))
    osm_seconds = perf_counter() - t
    manifest = dict(
        event_date=event,  # as supplied: a date stays a date (offset basis label)
        cutoff=_iso(limit),
        search_windows={
            "s1": [_iso(pre_start), _iso(s1_end)],
            "s2_pre": [_iso(pre_start), _iso(pre_limit)],
            "s2_post": [_iso(post_start), _iso(s2_end)],
        },
        bbox=list(geom.bounds),
        pre=scenes[:-1],
        post=scenes[-1],
        tracks=tracks,
        dem=dem,
        hand=hand,
        **optical,
        optical_candidates=candidates,
        osm=osm,
        rejected_scenes=rejected,
        synthetic=False,
        limitations=limitations,
        sentinel_hub=client.log,
        acquisition_timings_seconds={
            "satellite_discovery": discovery,
            "sar_download": sar_seconds,
            "dem_hand": terrain_seconds,
            "optical": optical_seconds,
            "osm": osm_seconds,
        },
    )
    manifest["pairing"] = pairing_provenance(manifest["pre"], manifest["post"])
    manifest["limitations"] += pairing_limitations(manifest["pairing"]) + track_notes
    key = cache_key(
        {"aoi": geom.wkt, "event": event, "cutoff": _iso(limit), "config": config}
    )
    path = folder / f"{key}-manifest.json"
    write_json(path, manifest)
    result = load_local(path)
    result["provider"] = "cdse"
    return result
