import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from pyproj import CRS, Transformer

from .common import inference_path, metric_crs, utc


def slope_degrees(dem, profile):
    dem = np.asarray(dem, dtype="float64")
    if not np.isfinite(dem).any():
        return np.full(dem.shape, np.nan)
    transform = profile["transform"]
    crs = CRS.from_user_input(profile["crs"])
    if (not crs.is_projected or transform.b or transform.d
            or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info)
            or min(dem.shape) < 2):
        raise ValueError("Slope requires a north-up projected metre grid")
    dx, dy = abs(transform.a), abs(transform.e)
    if crs.to_epsg() == 3857:
        x, y = transform * (dem.shape[1] / 2, dem.shape[0] / 2)
        _, latitude = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
        dx *= np.cos(np.deg2rad(latitude))
        dy *= np.cos(np.deg2rad(latitude))
    gy, gx = np.gradient(dem, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def align(path, profile, band=1, categorical=False):
    out = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(inference_path(path)) as src:
        data = src.read(band, masked=True).astype("float32").filled(np.nan)
        reproject(
            data,
            out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
    return out


def _model_pre_channels(arrays, scenes, vv, vh):
    if len(scenes) <= 2:
        return
    latest = sorted(range(len(scenes)), key=lambda index: utc(scenes[index]["timestamp"]))[-2:]
    # Keep the full VV reference/stack for hazard science; model training used two dates.
    for name, stack in (("pre_vv", vv), ("pre_vh", vh)):
        power = np.ma.masked_invalid(10 ** (stack[latest] / 10))
        arrays[name] = (10 * np.ma.log10(np.ma.median(power, axis=0))).filled(np.nan).astype("float32")


def prepare(manifest, geom, resolution, output, cache=None, model_inputs=False):
    crs = metric_crs(geom)
    projected = gpd.GeoSeries([geom], crs=4326).to_crs(crs).iloc[0]
    from rasterio.warp import calculate_default_transform

    for scene in [*manifest["pre"], manifest["post"]]:
        with rasterio.open(inference_path(scene["path"])) as src:
            native, _, _ = calculate_default_transform(
                src.crs, crs, src.width, src.height, *src.bounds
            )
            resolution = max(resolution, abs(native.a), abs(native.e))
    left, bottom, right, top = projected.bounds
    profile = dict(
        driver="GTiff",
        height=math.ceil((top - bottom) / resolution),
        width=math.ceil((right - left) / resolution),
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(left, top, resolution, resolution),
        nodata=np.nan,
        compress="deflate",
    )
    inside = geometry_mask(
        [projected],
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        invert=True,
    )
    cache_path = None
    if cache:
        inputs = {
            key: manifest[key]
            for key in (
                "pre",
                "post",
                "dem",
                "hand",
                "permanent_water",
                "optical_pre",
                "optical_post",
            )
            if manifest.get(key)
        }
        identity = json.dumps(
            {
                "version": "alignment-v3",  # pre-event stack (P04) + joint per-date SAR support (owner v2)
                "inputs": inputs,
                "aoi": geom.wkt,
                "resolution": resolution,
                "software_versions": manifest.get("software_versions", {}),
                **({"model_inputs": "latest-two-pre-v1"} if model_inputs else {}),
            },
            sort_keys=True,
            default=str,
        )
        cache_path = inference_path(
            Path(cache) / f"{hashlib.sha256(identity.encode()).hexdigest()}.npz"
        )
        if cache_path.exists():
            arrays = {}
            with np.load(cache_path, allow_pickle=False) as cached:
                for name in cached.files:
                    key, separator, band = name.partition("__")
                    if separator:
                        arrays.setdefault(key, {})[band] = cached[name]
                    else:
                        arrays[key] = cached[name]
            for key in ("pre", "post", "valid", "aoi"):
                if arrays[key].shape != inside.shape:
                    raise ValueError(
                        "Cached alignment dimensions do not match analysis grid"
                    )
                save_raster(
                    Path(output) / f"{key}.tif", arrays[key].astype("float32"), profile
                )
            manifest["processed_cache"] = {"path": str(cache_path), "hit": True}
            return arrays, profile
    pre_stack = np.stack([align(s["path"], profile) for s in manifest["pre"]])
    # Median in linear power; convert the robust reference back to dB.
    masked = np.ma.masked_invalid(10 ** (pre_stack / 10))
    pre = (
        (10 * np.ma.log10(np.ma.median(masked, axis=0)))
        .filled(np.nan)
        .astype("float32")
    )
    post = align(manifest["post"]["path"], profile)
    arrays = {
        "pre": pre,
        "pre_stack": pre_stack.astype("float32"),  # per-date VV dB (P04 persistence, z-score)
        "post": post,
        "aoi": inside,
        "valid": np.isfinite(pre) & np.isfinite(post) & inside,
    }
    with rasterio.open(inference_path(manifest["post"]["path"])) as src:
        has_vh = src.count >= 2 and manifest["post"]["polarization"] == "VV/VH"
    if has_vh:
        arrays["post_vh"] = align(manifest["post"]["path"], profile, 2)
        vh = np.stack([align(scene["path"], profile, 2) for scene in manifest["pre"]])
        arrays["pre_vh"] = (
            (
                10
                * np.ma.log10(
                    np.ma.median(np.ma.masked_invalid(10 ** (vh / 10)), axis=0)
                )
            )
            .filled(np.nan)
            .astype("float32")
        )
        if model_inputs:
            _model_pre_channels(arrays, manifest["pre"], pre_stack, vh)
    for scene in [*manifest["pre"], manifest["post"]]:
        if scene.get("processing", {}).get("backCoeff") == "GAMMA0_TERRAIN":
            arrays["valid"] &= np.isfinite(align(scene["path"], profile))
            if has_vh:
                arrays["valid"] &= np.isfinite(align(scene["path"], profile, 2))
            masks = scene.get("masks")
            if masks:
                bands = masks["bands"]
                arrays["valid"] &= align(masks["path"], profile, bands.index("dataMask") + 1, True) == 1
                arrays["valid"] &= align(masks["path"], profile, bands.index("shadowMask") + 1, True) == 0
            else:
                with rasterio.open(inference_path(scene["path"])) as source:
                    stacked_masks = source.count >= 4
                if stacked_masks:
                    arrays["valid"] &= align(scene["path"], profile, 3, True) == 1
                    arrays["valid"] &= align(scene["path"], profile, 4, True) == 0
                elif scene.get("provider") == "sentinelhub" or str(scene.get("source", "")).startswith("CDSE Sentinel Hub"):
                    raise ValueError("CDSE gamma0 requires per-date dataMask and shadowMask")
    for key in ("dem", "hand", "permanent_water"):
        if manifest.get(key):
            arrays[key] = align(
                manifest[key]["path"], profile, categorical=key == "permanent_water"
            )
    for key in ("optical_pre", "optical_post"):
        if manifest.get(key):
            scene = manifest[key]
            bands = scene.get(
                "bands", {"green": 1, "red": 2, "nir": 3, "swir": 4, "quality": 5}
            )
            arrays[key] = {
                name: align(scene["path"], profile, band, name == "quality")
                for name, band in bands.items()
            }
            scale = scene.get("reflectance_scale", 0.0001)
            for name in ("green", "red", "nir", "swir", "blue"):
                if name in arrays[key]:
                    arrays[key][name] *= scale
    if cache_path:
        flat = {}
        for key, value in arrays.items():
            if isinstance(value, dict):
                flat.update({f"{key}__{band}": array for band, array in value.items()})
            else:
                flat[key] = value
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **flat)
        temporary.replace(cache_path)
        manifest["processed_cache"] = {"path": str(cache_path), "hit": False}
    output = Path(output)
    for key in ("pre", "post", "valid", "aoi"):
        save_raster(output / f"{key}.tif", arrays[key].astype("float32"), profile)
    return arrays, profile


def prepare_track(track, profile, aoi, model_inputs=False):
    """SAR arrays of one additional track on an existing analysis grid (P04 fusion)."""
    pre_stack = np.stack([align(s["path"], profile) for s in track["pre"]])
    masked = np.ma.masked_invalid(10 ** (pre_stack / 10))
    pre = (10 * np.ma.log10(np.ma.median(masked, axis=0))).filled(np.nan).astype("float32")
    post = align(track["post"]["path"], profile)
    arrays = {
        "pre": pre,
        "pre_stack": pre_stack.astype("float32"),
        "post": post,
        "aoi": aoi,
        "valid": np.isfinite(pre) & np.isfinite(post) & aoi,
    }
    with rasterio.open(track["post"]["path"]) as src:
        if src.count >= 2 and track["post"]["polarization"] == "VV/VH":
            arrays["post_vh"] = align(track["post"]["path"], profile, 2)
            vh = np.stack([align(s["path"], profile, 2) for s in track["pre"]])
            arrays["pre_vh"] = (
                10 * np.ma.log10(np.ma.median(np.ma.masked_invalid(10 ** (vh / 10)), axis=0))
            ).filled(np.nan).astype("float32")
            if model_inputs:
                _model_pre_channels(arrays, track["pre"], pre_stack, vh)
    return arrays


def save_raster(path, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
