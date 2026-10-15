"""Terrain context and viewable SAR context, preserving explicit provenance."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .common import write_json


def satellite_context(path, output, scene):
    output = Path(output)
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 4326, src.width, src.height, *src.bounds
        )
        # Limit the display asset; source raster stays unchanged.
        factor = max(1, width / 1000, height / 1000)
        width, height = max(1, int(width / factor)), max(1, int(height / factor))
        transform = transform * rasterio.Affine.scale(factor, factor)
        target = np.full((height, width), np.nan, dtype="float32")
        reproject(
            src.read(1),
            target,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=4326,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    gray = np.clip((np.nan_to_num(target, nan=-30) + 30) / 30 * 255, 0, 255).astype(
        "uint8"
    )
    rgba = np.stack(
        [gray, gray, gray, np.where(np.isfinite(target), 255, 0).astype("uint8")],
        axis=-1,
    )
    Image.fromarray(rgba).save(output / "satellite_context.png")
    write_json(
        output / "satellite_context.json",
        {
            "bounds": list(rasterio.transform.array_bounds(height, width, transform)),
            "timestamp": scene["timestamp"],
            "source": scene.get("source", "local SAR"),
            "display": "VV dB grayscale [-30,0]; display resampling is not additional resolution",
        },
    )
