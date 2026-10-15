"""Rebuild mapped routes through the pinned upstream graph and snapping code."""

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
from pyproj import Transformer
from rasterio.io import MemoryFile

from backend.app.evidence.claims import read_bytes, read_json
from backend.app.schemas import HimaRoadRun, RouteResult
from backend.app.settings import Settings

MIN_LINK_LENGTH_M = 1e-8
MAX_RECONSTRUCTION_SETTLEMENTS = 50
MAX_RECONSTRUCTION_LINKS = 2000


class ReconstructionUnavailable(ValueError):
    """Reviewed output cannot faithfully reconstruct a graph."""


@dataclass
class Network:
    graph: Any
    sources: dict[str, Any]
    targets: dict[str, Any]
    removed: dict[str, frozenset[str]]
    settlements: dict[str, dict[str, Any]]
    assumptions: tuple[str, ...]


def removed_links(properties: list[dict[str, Any]]) -> dict[str, frozenset[str]]:
    """Reviewed legacy scenario definitions, independently expressed as set rules."""
    if any(not isinstance(p.get("link_id"), str) or not p["link_id"] for p in properties):
        raise ValueError("road link IDs must be nonempty strings")
    closed = {p["link_id"] for p in properties if p["operational_status"] == "confirmed_closed"}
    likely = {
        p["link_id"]
        for p in properties
        if p["operational_status"] != "confirmed_open" and p["evidence_category"] == "likely_impact"
    }
    worst = {
        p["link_id"]
        for p in properties
        if p["operational_status"] != "confirmed_open"
        and (
            p["operational_status"] == "unknown"
            or p["evidence_category"] in {"observed_change", "possible_impact", "unobservable"}
        )
    }
    return {
        "baseline": frozenset(),
        "1": frozenset(closed),
        "2": frozenset(closed | likely),
        "3": frozenset(closed | likely | worst),
    }


def scenario_removals(
    run: HimaRoadRun, properties: list[dict[str, Any]]
) -> dict[str, frozenset[str]]:
    if run.schema_version == "3.0":
        from backend.app.evidence.persisted import removed_sets

        return removed_sets(properties)
    return removed_links(properties)


def build_network(run: HimaRoadRun, settings: Settings, mode: str) -> Network:
    from himaroad.connectivity import build_graph  # pyright: ignore[reportMissingImports]

    if run.schema_version == "3.0":
        from himaroad.connectivity import snap_point  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

        def snap(point, graph, tolerance):
            return snap_point(graph, point, tolerance)
    else:
        from himaroad.connectivity import _snap as snap  # pyright: ignore[reportMissingImports]

    if run.status != "complete" or run.schema_version not in {None, "1.0", "3.0"}:
        raise ValueError("routes require a reviewed complete output contract")
    if mode not in {"vehicle", "pedestrian"}:
        raise ValueError("unknown routing mode")
    manifest = read_json(run, settings, "manifest.json")
    with MemoryFile(read_bytes(run, settings, "pre.tif")) as memory, memory.open() as raster:
        crs = raster.crs
    layers = {
        name: read_json(run, settings, name + ".geojson")["features"]
        for name in ("roads", "settlements", "destinations")
    }
    if run.schema_version == "3.0" and (
        len(layers["settlements"]) > MAX_RECONSTRUCTION_SETTLEMENTS
        or len(layers["roads"]) > MAX_RECONSTRUCTION_LINKS
    ):
        raise ReconstructionUnavailable(
            "Rebuilding exceeds the bounded interactive cross-check budget "
            "(50 settlements or 2000 links); persisted routes remain available, "
            "reopening gains unavailable."
        )
    frames = {}
    for name, features in layers.items():
        frames[name] = (
            gpd.GeoDataFrame.from_features(features, crs=4326).to_crs(crs)
            if features
            else gpd.GeoDataFrame(geometry=[], crs=crs)
        )
    roads = frames["roads"]
    if any(0 < g.length <= MIN_LINK_LENGTH_M for g in roads.geometry):
        raise ReconstructionUnavailable("near-zero road segment prevents graph reconstruction")
    zero_links = {str(row["link_id"]) for _, row in roads.iterrows() if row.geometry.length == 0}
    graph = build_graph(roads, mode)
    if any(e["link_id"] in zero_links for _, _, e in graph.edges(data=True)):
        raise ReconstructionUnavailable("zero-length input unexpectedly produced a graph edge")
    tolerance = manifest["configuration"]["snap_tolerance_m"]
    if type(tolerance) not in {int, float} or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid recorded snapping tolerance")
    # Persisted settlements are priority-sorted, whereas upstream snaps input order.
    # Do not infer the missing original order: flag reconstruction and compare parity.
    sources = {
        str(row["id"]): snap(row.geometry.representative_point(), graph, tolerance)
        for _, row in frames["settlements"].iterrows()
    }
    targets = {}
    for _, row in frames["destinations"].iterrows():
        # Vehicle upstream snaps every tier before selecting targets.
        # Foot upstream only snaps non-care destinations. Preserve that order.
        tier = row.get("destination_tier", "other")
        if (run.schema_version == "3.0" and tier == "healthcare_other") or (
            run.schema_version != "3.0"
            and mode == "pedestrian"
            and tier in {"primary_care", "healthcare_other"}
        ):
            continue
        node = snap(row.geometry.representative_point(), graph, tolerance)
        if tier not in {"primary_care", "healthcare_other"}:
            targets[str(row["id"])] = node
    assumptions = (
        "Mapped reachability only; operational usability is unverified.",
        "Hypothetical closures follow recorded HimaRoad scenario definitions.",
        "GeoJSON projection and snapping order may differ; producer flag parity is checked.",
        f"Snapping tolerance {tolerance} m; mapped network/destinations may be incomplete.",
    )
    if zero_links:
        assumptions += (
            "Zero-length links retained in provenance; pinned builder emitted no edges for them.",
        )
    return Network(
        graph,
        sources,
        targets,
        scenario_removals(run, [f["properties"] for f in layers["roads"]]),
        {f["properties"]["id"]: f["properties"] for f in layers["settlements"]},
        assumptions,
    )


def route(
    network: Network, settlement: str, scenario: str, reopen: frozenset[str] = frozenset()
) -> RouteResult:
    if scenario not in network.removed or settlement not in network.sources:
        raise ValueError("unknown scenario or settlement")
    removed = network.removed[scenario] - reopen
    base: dict[str, Any] = dict(
        settlement_id=settlement,
        scenario=scenario,
        target_id=None,
        cost_m=None,
        geometry=None,
        removed_link_ids=tuple(sorted(removed)),
        assumptions=network.assumptions,
    )
    origin = network.sources[settlement]
    targets = {k: v for k, v in network.targets.items() if v is not None}
    if origin is None or not targets:
        return RouteResult(**base, result="unassessable")
    graph: Any = nx.subgraph_view(
        network.graph, filter_edge=lambda a, b, k: network.graph[a][b][k]["link_id"] not in removed
    )
    distances, paths = nx.single_source_dijkstra(graph, origin, weight="length")  # type: ignore
    distances: Any
    paths: Any
    reachable = [
        (distances[node], identifier, node)
        for identifier, node in targets.items()
        if node in distances
    ]
    if not reachable:
        return RouteResult(**base, result="none")
    cost, target_id, destination = min(reachable, key=lambda item: (item[0], item[1]))
    nodes = paths[destination]
    coordinates = []
    for left, right in zip(nodes, nodes[1:], strict=False):
        edge = min(graph[left][right].values(), key=lambda e: (e["length"], e["link_id"]))
        segment = list(edge["geometry"].coords)
        start = graph.nodes[left]
        if (segment[-1][0] - start["x"]) ** 2 + (segment[-1][1] - start["y"]) ** 2 < (
            segment[0][0] - start["x"]
        ) ** 2 + (segment[0][1] - start["y"]) ** 2:
            segment.reverse()
        coordinates.extend(segment if not coordinates else segment[1:])
    if not coordinates:
        node = graph.nodes[origin]
        coordinates = [(node["x"], node["y"])] * 2
    project = Transformer.from_crs(graph.graph["crs"], 4326, always_xy=True).transform
    x, y = project([c[0] for c in coordinates], [c[1] for c in coordinates])
    geometry = {"type": "LineString", "coordinates": list(zip(x, y, strict=True))}
    return RouteResult(
        **(base | {"target_id": target_id, "cost_m": cost, "geometry": geometry}), result="found"
    )


def compute_routes(run: HimaRoadRun, settings: Settings) -> tuple[tuple[str, RouteResult], ...]:
    if run.status != "complete":
        return ()
    records = []
    for mode in ("vehicle", "pedestrian"):
        try:
            network = build_network(run, settings, mode)
        except ReconstructionUnavailable as exc:
            if run.status != "complete":
                return ()
            for feature in read_json(run, settings, "settlements.geojson")["features"]:
                for scenario in (
                    ("baseline", "1", "2", "3", "4")
                    if run.schema_version == "3.0"
                    else ("baseline", "1", "2", "3")
                ):
                    records.append(
                        (
                            mode,
                            RouteResult(
                                settlement_id=feature["properties"]["id"],
                                scenario=scenario,
                                target_id=None,
                                cost_m=None,
                                geometry=None,
                                result="unassessable",
                                removed_link_ids=(),
                                assumptions=(str(exc),),
                            ),
                        )
                    )
            continue
        for settlement in network.sources:
            for scenario in network.removed:
                result = route(network, settlement, scenario)
                expected = None
                props = network.settlements[settlement]
                if mode == "vehicle":
                    expected = props[
                        "baseline_accessible"
                        if scenario == "baseline"
                        else f"scenario{scenario}_accessible"
                    ]
                elif scenario == "2":
                    expected = props["pedestrian_accessible"]
                if expected is not None and expected != (result.result == "found"):
                    result = result.model_copy(
                        update={
                            "result": "unassessable",
                            "target_id": None,
                            "cost_m": None,
                            "geometry": None,
                            "assumptions": (
                                *result.assumptions,
                                "Rebuilt graph disagrees with producer accessibility flag.",
                            ),
                        }
                    )
                records.append((mode, result))
    return tuple(records)
