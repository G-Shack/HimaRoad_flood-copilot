"""Height above nearest drainage from a Copernicus DEM (soft flag only, D32/HR-X-001).

The DEM is a DSM and not hydrologically conditioned, so it is conditioned first:
priority-flood depression filling with a small epsilon (Barnes et al. 2014,
doi:10.1016/j.cageo.2013.04.024) leaves every inland cell a strictly lower
neighbour; D8 steepest descent on that surface gives each cell its receiver.
Streams are cells whose upstream area reaches a threshold; HAND is a cell's
height above the stream cell it drains to. Cells that drain off the grid without
meeting a stream stay unknown (NaN), never zero.
"""

import heapq
from pathlib import Path

import numpy as np
import rasterio

from .common import is_metre_crs

STREAM_AREA_KM2 = 1.0
HAND_VERSION = "hand-v1"  # bump when the algorithm changes (cache keys use it)
EPSILON_M = 1e-3
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def condition(dem: np.ndarray, epsilon: float = EPSILON_M):
    """Filled DEM, receiver index per cell (-1 = outlet) and processing order."""
    rows, cols = dem.shape
    valid = np.isfinite(dem)
    filled = np.where(valid, dem, np.nan).astype("float64")
    receiver = np.full(dem.size, -1, dtype=np.int64)
    done = ~valid.copy()
    heap = []
    # Outlets: valid cells on the grid edge or next to nodata.
    padded = np.pad(valid, 1, constant_values=False)
    interior = np.ones_like(valid)
    for dr, dc in NEIGHBOURS:
        interior &= padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    for index in np.flatnonzero(valid & ~interior):
        heapq.heappush(heap, (filled.flat[index], index))
        done.flat[index] = True
    order = []
    while heap:
        level, index = heapq.heappop(heap)
        order.append(index)
        r, c = divmod(index, cols)
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not done[nr, nc]:
                done[nr, nc] = True
                neighbour = nr * cols + nc
                filled.flat[neighbour] = max(filled.flat[neighbour], level + epsilon)
                receiver[neighbour] = index
                heapq.heappush(heap, (filled.flat[neighbour], neighbour))
    return filled, receiver, np.asarray(order, dtype=np.int64)


def d8_receivers(filled: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Steepest-descent neighbour per cell on the filled surface (-1 = outlet/nodata)."""
    rows, cols = filled.shape
    padded = np.pad(filled, 1, constant_values=np.nan)
    best = np.zeros(filled.shape)
    receiver = np.full(filled.shape, -1, dtype=np.int64)
    index = np.arange(filled.size).reshape(filled.shape)
    padded_index = np.pad(index, 1, constant_values=-1)
    for dr, dc in NEIGHBOURS:
        neighbour = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        drop = (filled - neighbour) / (cell_size * np.hypot(dr, dc))
        better = np.isfinite(drop) & (drop > best)
        best = np.where(better, drop, best)
        receiver = np.where(better, padded_index[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols], receiver)
    return receiver.ravel()


def hand_from_dem(dem: np.ndarray, cell_area_m2: float, stream_area_km2=STREAM_AREA_KM2):
    """HAND in metres (NaN where no stream is reached) and the stream mask."""
    filled, _, _ = condition(dem)
    receiver = d8_receivers(filled, np.sqrt(cell_area_m2))
    valid = np.flatnonzero(np.isfinite(filled.ravel()))
    # Receivers are strictly lower on the epsilon-filled surface, so sorting by
    # height orders every cell after its receiver.
    order = valid[np.argsort(filled.ravel()[valid], kind="stable")]
    accumulation = np.zeros(dem.size)
    accumulation[order] = 1.0
    for index in order[::-1]:
        if receiver[index] >= 0:
            accumulation[receiver[index]] += accumulation[index]
    stream = accumulation * cell_area_m2 >= stream_area_km2 * 1e6
    drain = np.full(dem.size, -1, dtype=np.int64)
    for index in order:
        if stream[index]:
            drain[index] = index
        elif receiver[index] >= 0:
            drain[index] = drain[receiver[index]]
    flat = dem.ravel()
    hand = np.where(drain >= 0, flat - flat[np.maximum(drain, 0)], np.nan)
    # A cell below its drainage point (DSM noise) is at the channel, not underground.
    hand = np.where(np.isfinite(hand), np.maximum(hand, 0.0), np.nan)
    return hand.reshape(dem.shape), stream.reshape(dem.shape)


def derive_hand(dem_path: Path, out_path: Path, stream_area_km2=STREAM_AREA_KM2) -> dict:
    """Write a HAND GeoTIFF on the DEM grid and return its provenance."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = src.profile | dict(dtype="float32", nodata=np.nan, count=1)
        if not is_metre_crs(src.crs):
            raise ValueError("HAND needs a DEM on a projected grid in metres")
        cell_area = abs(src.transform.a * src.transform.e)
    hand, stream = hand_from_dem(dem, cell_area, stream_area_km2)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(hand.astype("float32"), 1)
    return {
        "path": str(out_path),
        "source": "derived from Copernicus GLO-30 (depression-filled, D8)",
        "method": "priority-flood fill with epsilon (Barnes 2014), D8 steepest descent, "
        f"streams at >= {stream_area_km2} km2 upstream area",
        "stream_area_km2": stream_area_km2,
        "unknown_fraction": float(np.mean(~np.isfinite(hand[np.isfinite(dem)])))
        if np.isfinite(dem).any()
        else 1.0,
        "stream_cells": int(stream.sum()),
    }
