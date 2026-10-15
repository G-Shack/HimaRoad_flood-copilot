"""Settlements from OSM place nodes plus building clusters (T03.04, HR-C-006).

Clustering rule (DBSCAN, Ester et al. 1996, on building centroids in a metric CRS):
- a centroid is a *core* point when at least ``min_buildings`` centroids, itself
  included, lie within ``eps_m`` (distance <= eps_m);
- core points within ``eps_m`` of each other are connected; each connected
  component of core points is one cluster;
- a non-core centroid within ``eps_m`` of a core point is a *border* point and
  joins the cluster of its nearest core point (ties: lowest row); every other
  centroid is noise.
This equals scikit-learn ``DBSCAN(eps, min_samples=min_buildings)`` except that
border ties are resolved by distance instead of visiting order.

A cluster footprint is the convex hull of its member buildings buffered by
``eps_m``. A place node inside a footprint absorbs that cluster; clusters without
a place node become settlements of their own. Names come only from OSM place
nodes; a cluster with no place node within ``name_radius_m`` stays unnamed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import STRtree

from .common import is_metre_crs

CLUSTER_COLUMNS = ["cluster_id", "building_count", "building_ids", "geometry", "footprint"]


def _metric(frame, what):
    if not is_metre_crs(frame.crs):
        raise ValueError(f"{what} must be in a projected CRS in metres")


def _xy(geometries):
    centroids = geometries.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])


def dbscan_labels(xy: np.ndarray, eps_m: float, min_buildings: int) -> np.ndarray:
    """Cluster label per point (-1 = noise), using the rule in the module docstring."""
    labels = np.full(len(xy), -1, dtype=np.int64)
    if not len(xy):
        return labels
    tree = cKDTree(xy)
    core = tree.query_ball_point(xy, eps_m, return_length=True) >= min_buildings
    core_index = np.flatnonzero(core)
    if not len(core_index):
        return labels
    core_tree = cKDTree(xy[core_index])
    pairs = core_tree.query_pairs(eps_m, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(core_index),) * 2
    )
    _, component = connected_components(graph, directed=False)
    labels[core_index] = component
    others = np.flatnonzero(~core)
    if len(others):
        # cKDTree's bound is strict; nextafter makes it "<= eps" as documented (audit L1).
        distance, nearest = core_tree.query(xy[others], distance_upper_bound=np.nextafter(eps_m, np.inf))
        border = np.isfinite(distance)
        labels[others[border]] = component[nearest[border]]
    return labels


def building_clusters(buildings: gpd.GeoDataFrame, eps_m: float = 50.0, min_buildings: int = 5) -> gpd.GeoDataFrame:
    """Density clusters of building centroids, ordered west to east (then south to north)."""
    _metric(buildings, "Buildings")
    xy = _xy(buildings.geometry)
    labels = dbscan_labels(xy, eps_m, min_buildings)
    ids = buildings["id"].astype(str).to_numpy()
    rows = []
    for label in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == label)
        centre = xy[members].mean(axis=0)
        hull = buildings.geometry.iloc[members].union_all().convex_hull
        rows.append(
            {
                "building_count": len(members),
                "building_ids": sorted(ids[members].tolist()),
                "geometry": gpd.points_from_xy([centre[0]], [centre[1]])[0],
                "footprint": hull.buffer(eps_m),
                "_key": (round(centre[0], 3), round(centre[1], 3)),
            }
        )
    rows.sort(key=lambda row: (row["_key"], row["building_ids"][0]))
    for n, row in enumerate(rows):
        del row["_key"]
        row["cluster_id"] = f"cluster/{n}"
    frame = gpd.GeoDataFrame(
        rows or {column: [] for column in CLUSTER_COLUMNS}, geometry="geometry", crs=buildings.crs
    )[CLUSTER_COLUMNS]
    frame.attrs["clustering"] = {"method": "DBSCAN on building centroids", "eps_m": eps_m, "min_buildings": min_buildings}
    return frame


def _text(value):
    return value if isinstance(value, str) and value else None


def _place_cluster(places, clusters):
    """Cluster id each place node falls in (nearest cluster centre if footprints overlap), else None."""
    result = pd.Series([None] * len(places), index=places.index, dtype=object)
    if places.empty or clusters.empty:
        return result
    tree = STRtree(clusters["footprint"].to_numpy())
    place_index, cluster_index = tree.query(places.geometry.to_numpy(), predicate="within")
    best = {}
    for p, c in zip(place_index, cluster_index):
        distance = places.geometry.iloc[p].distance(clusters.geometry.iloc[c])
        if p not in best or distance < best[p][0]:
            best[p] = (distance, c)
    for p, (_, c) in best.items():
        result.iloc[p] = clusters["cluster_id"].iloc[c]
    return result


def merge_settlements(places: gpd.GeoDataFrame, clusters: gpd.GeoDataFrame, name_radius_m: float = 500.0) -> gpd.GeoDataFrame:
    """Place nodes plus building clusters that contain no place node.

    If several place nodes fall in one cluster, the one nearest the cluster centre
    (ties: lowest id) represents it and carries its ``building_count``; the others
    keep ``cluster_id`` with ``building_count`` 0, so counts are not duplicated.
    """
    _metric(places, "Place nodes")
    if not clusters.empty and clusters.crs != places.crs:
        raise ValueError("Place nodes and clusters must share a CRS")
    nodes = places.copy()
    nodes["id"] = nodes["id"].astype(str)
    nodes["name"] = [_text(name) for name in nodes.get("name", pd.Series(index=nodes.index))]
    nodes["kind"] = "place_node"
    nodes["name_source"] = ["osm_name" if name else "none" for name in nodes["name"]]
    nodes["cluster_id"] = _place_cluster(nodes, clusters)
    nodes["represents_cluster"] = False
    nodes["building_count"] = 0
    counts = dict(zip(clusters["cluster_id"], clusters["building_count"])) if not clusters.empty else {}
    centres = dict(zip(clusters["cluster_id"], clusters.geometry)) if not clusters.empty else {}
    for cluster_id, group in nodes[nodes["cluster_id"].notna()].groupby("cluster_id"):
        order = sorted(group.index, key=lambda i: (group.geometry[i].distance(centres[cluster_id]), group["id"][i]))
        nodes.loc[order[0], ["represents_cluster", "building_count"]] = [True, counts[cluster_id]]
    absorbed = set(nodes["cluster_id"].dropna())
    rest = clusters[~clusters["cluster_id"].isin(absorbed)] if not clusters.empty else clusters
    added = []
    named = nodes[nodes["name"].notna()]
    tree = cKDTree(_xy(named.geometry)) if len(named) else None
    for _, cluster in rest.iterrows():
        name, source = None, "none"
        if tree is not None:
            distance, nearest = tree.query([cluster.geometry.x, cluster.geometry.y], distance_upper_bound=name_radius_m)
            if np.isfinite(distance):
                name, source = f"near {named['name'].iloc[nearest]}", "nearest_place"
        added.append(
            {
                "id": cluster["cluster_id"],
                "name": name,
                "kind": "building_cluster",
                "name_source": source,
                "cluster_id": cluster["cluster_id"],
                "represents_cluster": True,
                "building_count": int(cluster["building_count"]),
                "geometry": cluster.geometry,
            }
        )
    frames = [nodes]
    if added:
        frames.append(gpd.GeoDataFrame(added, geometry="geometry", crs=places.crs))
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=places.crs)
    merged["building_count"] = merged["building_count"].astype(int)
    return merged


def assign_buildings(buildings, settlements, clusters, max_distance_m: float = 500.0) -> pd.Series:
    """Settlement id per building: its cluster's representative, else the nearest settlement within max_distance_m, else None."""
    _metric(buildings, "Buildings")
    result = pd.Series([None] * len(buildings), index=buildings.index, dtype=object, name="settlement_id")
    if settlements.empty:
        return result
    representative = dict(
        zip(
            settlements.loc[settlements["represents_cluster"].astype(bool), "cluster_id"],
            settlements.loc[settlements["represents_cluster"].astype(bool), "id"],
        )
    )
    member = {}
    for _, cluster in clusters.iterrows():
        if cluster["cluster_id"] not in representative:
            raise ValueError(f"{cluster['cluster_id']} has no representative settlement")
        for building_id in cluster["building_ids"]:
            member[building_id] = representative[cluster["cluster_id"]]
    ids = buildings["id"].astype(str)
    in_cluster = ids.isin(member.keys()).to_numpy()
    result[in_cluster] = ids[in_cluster].map(member).to_numpy()
    loose = np.flatnonzero(~in_cluster)
    if len(loose):
        tree = cKDTree(_xy(settlements.geometry))
        distance, nearest = tree.query(_xy(buildings.geometry.iloc[loose]), distance_upper_bound=max_distance_m)
        found = np.isfinite(distance)
        result.iloc[loose[found]] = settlements["id"].astype(str).to_numpy()[nearest[found]]
    return result


def cluster_sensitivity(buildings, eps_values=(25, 50, 100), min_values=(3, 5, 10), places=None) -> list[dict]:
    """Cluster counts per (eps_m, min_buildings); with ``places``, also the clusters that would become new settlements."""
    rows = []
    for eps_m in eps_values:
        for min_buildings in min_values:
            clusters = building_clusters(buildings, eps_m, min_buildings)
            clustered = int(clusters["building_count"].sum()) if not clusters.empty else 0
            row = {
                "eps_m": eps_m,
                "min_buildings": min_buildings,
                "clusters": len(clusters),
                "clustered_buildings": clustered,
                "noise_buildings": len(buildings) - clustered,
                "largest_cluster": int(clusters["building_count"].max()) if not clusters.empty else 0,
            }
            if places is not None:
                merged = merge_settlements(places, clusters)
                new = merged["kind"] == "building_cluster"
                row["new_settlements"] = int(new.sum())
                row["new_named_near_place"] = int((new & (merged["name_source"] == "nearest_place")).sum())
            rows.append(row)
    return rows
