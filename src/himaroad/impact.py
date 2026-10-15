"""Satellite evidence categories kept separate from operational status."""

import geopandas as gpd
import numpy as np
from rasterio import Affine
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.ops import substring

from .connectivity import _metric, _tag


def segment_roads(roads, max_length=100, min_segment_length_m=0.01):
    """Split ways into segments of at most max_length metres.

    Every vertex stays a cut, so OSM node topology is never lost. A regular
    max_length cut closer than min_segment_length_m to a vertex is dropped,
    so no sub-tolerance sliver is created. Ways where that happened, or that
    contain identical consecutive coordinates, are listed in
    attrs["degenerate_segments_dropped"].
    """
    _metric(roads)
    if max_length <= 0:
        raise ValueError("Road segment length must be positive")
    if not 0 <= min_segment_length_m < max_length:
        raise ValueError("min_segment_length_m must be in [0, max_length)")
    records = []
    dropped = []
    for _, row in roads.iterrows():
        line = row.geometry
        if line is None or line.is_empty or line.geom_type != "LineString":
            continue
        coordinates = np.asarray(line.coords)[:, :2]
        positions = np.r_[
            0, np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))
        ]
        positions[-1] = line.length
        vertices = {float(distance): i for i, distance in enumerate(positions)}
        anchors = np.array(sorted(vertices))
        regular = [
            float(d)
            for d in np.arange(0, line.length, max_length)
            if np.abs(anchors - d).min() > min_segment_length_m
        ]
        if len(regular) < len(np.arange(0, line.length, max_length)) - 1:
            dropped.append(str(row["id"]))  # 0 always coincides with the first vertex.
        cuts = sorted(set(vertices) | set(regular))
        ids = row.get("node_ids")
        ids = (
            ids
            if isinstance(ids, (list, tuple)) and len(ids) == len(line.coords)
            else None
        )
        number = 0
        for start, end in zip(cuts, cuts[1:]):
            segment = substring(line, start, end)
            if segment.geom_type != "LineString" or len(set(segment.coords)) < 2:
                dropped.append(str(row["id"]))  # identical consecutive coordinates
                continue
            record = row.to_dict()
            record["osm_id"] = str(row.get("osm_id", row["id"]))
            record["id"] = record["link_id"] = f"{row['id']}:segment/{number}"
            record["geometry"] = segment
            if ids:
                record["node_ids"] = [
                    ids[vertices[d]]
                    if d in vertices
                    else f"{row['id']}:position/{d:.6f}"
                    for d in (start, end)
                ]
            records.append(record)
            number += 1
        if not number:
            dropped.append(str(row["id"]))
    frame = (
        gpd.GeoDataFrame(records, geometry="geometry", crs=roads.crs)
        if records
        else roads.copy()
    )
    frame.attrs["degenerate_segments_dropped"] = sorted(set(dropped))
    frame.attrs["degenerate_tolerance_m"] = min_segment_length_m
    return frame


def feature_mask(area, transform, shape):
    """all_touched mask of one geometry, rasterised only in its padded pixel window.

    Identical to rasterising on the full grid (pixels off the grid are never
    counted), but proportional to the feature's size rather than the AOI's.
    """
    if area.is_empty:  # no pixels, as full-grid rasterisation gives (review L3)
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), bool)
    minx, miny, maxx, maxy = area.bounds
    inverse = ~transform
    xs, ys = zip(*(inverse @ corner for corner in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]))
    r0, r1 = max(int(np.floor(min(ys))) - 1, 0), min(int(np.ceil(max(ys))) + 1, shape[0])
    c0, c1 = max(int(np.floor(min(xs))) - 1, 0), min(int(np.ceil(max(xs))) + 1, shape[1])
    rows, cols = slice(r0, max(r0, r1)), slice(c0, max(c0, c1))
    if r1 <= r0 or c1 <= c0:
        return rows, cols, np.zeros((max(0, r1 - r0), max(0, c1 - c0)), bool)
    selected = geometry_mask(
        [mapping(area)],
        out_shape=(r1 - r0, c1 - c0),
        transform=transform @ Affine.translation(c0, r0),
        invert=True,
        all_touched=True,
    )
    return rows, cols, selected


def assess(
    infrastructure,
    flood,
    disturbance,
    valid,
    transform,
    observation_timestamp,
    buffer_m=10,
):
    _metric(infrastructure)
    if buffer_m < 0:
        raise ValueError("buffer_m must be nonnegative")
    flood, disturbance, valid = (
        np.asarray(flood),
        np.asarray(disturbance),
        np.asarray(valid, dtype=bool),
    )
    if (
        flood.ndim != 2
        or flood.shape != disturbance.shape
        or flood.shape != valid.shape
    ):
        raise ValueError("Evidence arrays must share a two-dimensional grid")
    result = infrastructure.copy()
    records = []
    for _, feature in infrastructure.iterrows():
        geometry = feature.geometry
        area = (
            geometry.buffer(buffer_m)
            if geometry.geom_type not in {"Polygon", "MultiPolygon"}
            else geometry
        )
        rows, cols, selected = feature_mask(area, transform, valid.shape)
        observed = selected & valid[rows, cols]
        count = int(selected.sum())
        coverage = int(observed.sum()) / count if count else 0.0
        water = bool(np.any(observed & (flood[rows, cols] > 0)))
        change = bool(np.any(observed & (disturbance[rows, cols] > 0)))
        category = (
            "observed_change"
            if water or change
            else "unobservable"
            if coverage < 0.8
            else "no_detected_change"
        )
        status = _tag(feature, "operational_status", "unknown")
        source = _tag(feature, "independent_source")
        if status not in {"confirmed_open", "confirmed_closed"} or not source:
            status = "unknown"
        records.append(
            dict(
                evidence_category=category,
                operational_status=status,
                physical_condition="unknown",
                valid_observation_fraction=coverage,
                flood_intersection=water,
                disturbance_intersection=change,
                potentially_exposed=water or change,
                observation_timestamp=str(observation_timestamp),
            )
        )
    for key in (
        records[0]
        if records
        else [
            "evidence_category",
            "operational_status",
            "physical_condition",
            "valid_observation_fraction",
            "flood_intersection",
            "disturbance_intersection",
            "potentially_exposed",
            "observation_timestamp",
        ]
    ):
        result[key] = [record[key] for record in records]
    return result
