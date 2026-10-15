"""SAR viewing-geometry masks (T04.01, HR-C-008): layover and radar shadow from the DEM.

Sentinel Hub supplies a shadowMask and the local incidence angle but no layover mask, and
its incidence angle is unsigned, so layover cannot be read from it. Here the look direction
and incidence angle are fitted to the provider's local incidence angle (or taken from the
nominal Sentinel-1 orbit when it is absent), then active and passive layover and shadow are
traced along each ground-range line of the Copernicus DEM. Parallel rays (flat-Earth,
far-field) are assumed; across a 20 km AOI the incidence angle varies by about 1 degree.
"""

import math

import numpy as np

S1_INCLINATION_DEG = 98.18
NOMINAL_INCIDENCE_DEG = 39.0  # IW mid-swath (29-46 degrees across the swath)
MAX_TRACE_M = 8000.0


def nominal_look_azimuth(pass_, lat):
    """Right-looking look azimuth (degrees clockwise from north) from the orbit alone.

    Ground-track heading from the inclination, ignoring Earth rotation (a few degrees).
    """
    ratio = math.cos(math.radians(S1_INCLINATION_DEG)) / max(math.cos(math.radians(lat)), 1e-6)
    heading = math.degrees(math.asin(max(-1.0, min(1.0, ratio))))  # ascending, from north
    if pass_ == "DESCENDING":
        heading = 180.0 - heading
    elif pass_ != "ASCENDING":
        raise ValueError("Pass must be ASCENDING or DESCENDING")
    return (heading + 90.0) % 360.0


def gradients(dem, res):
    """dz/dEast and dz/dNorth on a north-up grid (rows run south)."""
    dz_drow, dz_dcol = np.gradient(dem.astype("float64"), res, res)
    return dz_dcol, -dz_drow


def _incidence_from_gradients(de, dn, look_azimuth, incidence):
    a, t = math.radians(look_azimuth), math.radians(incidence)
    # Unit vector from the ground towards the sensor.
    to_sensor = (-math.sin(a) * math.sin(t), -math.cos(a) * math.sin(t), math.cos(t))
    cosine = (-de * to_sensor[0] - dn * to_sensor[1] + to_sensor[2]) / np.sqrt(de**2 + dn**2 + 1)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def local_incidence(dem, res, look_azimuth, incidence):
    """Unsigned angle between the surface normal and the direction to the sensor (degrees)."""
    return _incidence_from_gradients(*gradients(dem, res), look_azimuth, incidence)


def fit_geometry(dem, res, provider_lia, provider_shadow, pass_, lat, max_pixels=200_000):
    """Look azimuth and incidence angle that best reproduce the provider's local incidence.

    Physics only: the provider angle is computed from the Copernicus DEM, so the fit
    recovers the acquisition geometry. Falls back to the nominal orbit when unusable.
    Slopes come from the full-resolution DEM; only the compared pixels are subsampled.
    """
    nominal = nominal_look_azimuth(pass_, lat)
    usable = None
    if provider_lia is not None:
        de, dn = gradients(dem, res)
        usable = (
            np.isfinite(de) & np.isfinite(dn) & np.isfinite(provider_lia)
            & (np.nan_to_num(provider_shadow, nan=1) == 0)
        )
    if usable is None or int(usable.sum()) < 500:
        return {
            "look_azimuth_deg": round(nominal, 2),
            "incidence_deg": NOMINAL_INCIDENCE_DEG,
            "method": "nominal Sentinel-1 orbit (provider local incidence angle unavailable)",
            "median_abs_error_deg": None,
            "correlation": None,
            "pixels": 0,
        }
    index = np.flatnonzero(usable)
    if len(index) > max_pixels:  # Deterministic thinning keeps the search fast.
        index = index[:: int(math.ceil(len(index) / max_pixels))]
    de, dn, target = de.ravel()[index], dn.ravel()[index], provider_lia.ravel()[index]

    def error(azimuth, incidence):
        return float(np.median(np.abs(_incidence_from_gradients(de, dn, azimuth, incidence) - target)))

    _, az0, inc0 = min(
        (error(az, inc), az, inc) for az in np.arange(nominal - 15, nominal + 15.1, 1.0)
        for inc in np.arange(29.0, 46.1, 1.0)
    )
    err, azimuth, incidence = min(
        (error(az, inc), az, inc) for az in np.arange(az0 - 1, az0 + 1.01, 0.1)
        for inc in np.arange(inc0 - 1, inc0 + 1.01, 0.1)
    )
    model = _incidence_from_gradients(de, dn, azimuth, incidence)
    return {
        "look_azimuth_deg": round(float(azimuth % 360), 2),
        "incidence_deg": round(float(incidence), 2),
        "method": "fitted to the provider local incidence angle (Copernicus DEM, non-shadow pixels)",
        "median_abs_error_deg": round(err, 3),
        "correlation": round(float(np.corrcoef(model, target)[0, 1]), 4),
        "pixels": int(len(index)),
    }


def _shift(values, drow, dcol):
    """values[r + drow, c + dcol] at (r, c); NaN outside the grid."""
    out = np.full(values.shape, np.nan, dtype=values.dtype)
    h, w = values.shape
    rs, re = max(0, -drow), min(h, h - drow)
    cs, ce = max(0, -dcol), min(w, w - dcol)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = values[rs + drow:re + drow, cs + dcol:ce + dcol]
    return out


def trace_reach_m(dem, incidence):
    """Farthest ground distance at which this DEM's relief can cause layover or shadow."""
    relief = float(np.nanmax(dem) - np.nanmin(dem)) if np.isfinite(dem).any() else 0.0
    t = math.radians(incidence)
    return max(relief / math.tan(t), relief * math.tan(t))


def geometry_masks(dem, res, look_azimuth, incidence, max_trace_m=MAX_TRACE_M):
    """Active and passive layover and radar shadow traced along ground-range lines.

    With x the ground distance away from the sensor and z the height:
    - slant range r = x sin(t) - z cos(t): a pixel is in layover when a farther pixel on its
      range line has r no larger (or a nearer one has r no smaller), i.e. echoes overlap;
    - s = x cos(t) + z sin(t): a pixel is in shadow when a nearer pixel has a larger s,
      i.e. terrain between it and the sensor rises above the ray.
    """
    dem = dem.astype("float64")
    a, t = math.radians(look_azimuth), math.radians(incidence)
    rows, cols = np.indices(dem.shape)
    x = (cols * res) * math.sin(a) + (-rows * res) * math.cos(a)
    r = x * math.sin(t) - dem * math.cos(t)
    s = x * math.cos(t) + dem * math.sin(t)
    reach = min(max_trace_m, trace_reach_m(dem, incidence))
    steps = int(math.ceil(reach / res)) + 1
    layover = np.zeros(dem.shape, bool)
    shadow = np.zeros(dem.shape, bool)
    seen = set()
    for k in range(1, steps + 1):
        drow, dcol = int(round(-k * math.cos(a))), int(round(k * math.sin(a)))
        if (drow, dcol) in seen:
            continue
        seen.add((drow, dcol))
        farther_r = _shift(r, drow, dcol)
        nearer_r = _shift(r, -drow, -dcol)
        nearer_s = _shift(s, -drow, -dcol)
        with np.errstate(invalid="ignore"):
            layover |= (farther_r <= r) | (nearer_r >= r)
            shadow |= nearer_s > s
    unknown = ~np.isfinite(dem)
    layover[unknown] = False
    shadow[unknown] = False
    return layover, shadow


def _buffered(profile, pixels):
    transform = profile["transform"]
    return profile | {
        "height": profile["height"] + 2 * pixels,
        "width": profile["width"] + 2 * pixels,
        "transform": transform * transform.translation(-pixels, -pixels),
    }


def viewing_masks(manifest, profile, aoi, buffer_m=5000.0):
    """Layover and shadow on the analysis grid for one run, with provenance.

    The provider shadowMask counts on any date of the same-track stack. Geometric layover
    and shadow are traced on a DEM buffered around the AOI, because terrain outside the AOI
    can overlay or shadow pixels inside it.
    """
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    from .preprocess import align

    shape = (profile["height"], profile["width"])
    scenes = [*manifest["pre"], manifest["post"]]
    limitations = []
    provider_shadow = np.zeros(shape, bool)
    provider_lia = provider_flag = None
    with_masks = [scene for scene in scenes if scene.get("masks")]
    for scene in with_masks:
        path = scene["masks"]["path"]
        flag = align(path, profile, 2, categorical=True)
        provider_shadow |= flag == 1
        if scene is manifest["post"]:
            provider_flag = flag
            provider_lia = align(path, profile, 3)
    if len(with_masks) < len(scenes):
        limitations.append(
            f"Provider shadowMask unavailable for {len(scenes) - len(with_masks)} of {len(scenes)} "
            "SAR scenes; shadow there relies on DEM geometry only."
        )
    result = {
        "provider_shadow": provider_shadow,
        "layover": np.zeros(shape, bool),
        "geometric_shadow": np.zeros(shape, bool),
    }
    stats = {"masks_version": "sar-geometry-v1", "geometry": None}
    pass_ = manifest["post"].get("pass")
    dem = manifest.get("dem")
    if not dem or pass_ not in ("ASCENDING", "DESCENDING"):
        limitations.append(
            "Layover and geometric shadow not computed: "
            + ("no DEM." if not dem else "orbit pass unknown.")
            + " Only the provider shadowMask (if any) is applied."
        )
    else:
        pixels = int(math.ceil(min(buffer_m, MAX_TRACE_M) / abs(profile["transform"].a)))
        wide = _buffered(profile, pixels)
        elevation = align(dem["path"], wide)
        inner = (slice(pixels, pixels + shape[0]), slice(pixels, pixels + shape[1]))
        rows, cols = shape[0] // 2, shape[1] // 2
        x, y = xy(profile["transform"], rows, cols)
        lat = warp_transform(profile["crs"], "EPSG:4326", [x], [y])[1][0]
        geometry = fit_geometry(
            elevation[inner], abs(profile["transform"].a), provider_lia, provider_flag, pass_, lat
        )
        layover, shadow = geometry_masks(
            elevation, abs(profile["transform"].a), geometry["look_azimuth_deg"],
            geometry["incidence_deg"],
        )
        result["layover"], result["geometric_shadow"] = layover[inner], shadow[inner]
        buffer_used = pixels * abs(profile["transform"].a)
        reach = trace_reach_m(elevation, geometry["incidence_deg"])
        stats["geometry"] = geometry | {
            "pass": pass_,
            "dem_buffer_m": buffer_used,
            "trace_reach_m": round(reach, 1),
            "max_trace_m": MAX_TRACE_M,
        }
        if reach > min(buffer_used, MAX_TRACE_M):
            limitations.append(
                f"The DEM relief can cause layover/shadow up to {reach / 1000:.1f} km away, beyond the "
                f"{min(buffer_used, MAX_TRACE_M) / 1000:.1f} km traced around the AOI: terrain farther "
                "out may lay over or shadow AOI-edge pixels without being masked."
            )
        if geometry["median_abs_error_deg"] is None:
            limitations.append(
                "Layover/shadow geometry uses the nominal Sentinel-1 orbit (no provider local "
                "incidence angle): look direction and incidence are approximate."
            )
    masked = aoi & (result["layover"] | result["geometric_shadow"] | provider_shadow)
    area = max(1, int(aoi.sum()))
    stats |= {
        "layover_percent": round(100 * float((aoi & result["layover"]).sum()) / area, 2),
        "geometric_shadow_percent": round(100 * float((aoi & result["geometric_shadow"]).sum()) / area, 2),
        "provider_shadow_percent": round(100 * float((aoi & provider_shadow).sum()) / area, 2),
        "masked_percent": round(100 * float(masked.sum()) / area, 2),
        "rule": "unobservable where geometric layover, geometric shadow, or the provider "
        "shadowMask on any date of the stack",
    }
    result |= {"masked": masked, "stats": stats, "limitations": limitations}
    return result
