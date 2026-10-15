"""Mapped network accessibility under explicit disruption assumptions."""

import math
from collections.abc import Mapping

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree


def _metric(frame):
    if (
        frame.crs is None
        or not CRS(frame.crs).is_projected
        or any(a.unit_conversion_factor != 1 for a in CRS(frame.crs).axis_info[:2])
    ):
        raise ValueError("A projected CRS in metres is required")


def _tag(row, key, default=""):
    value = row.get(key, default)
    return (
        default
        if value is None or (isinstance(value, float) and math.isnan(value))
        else str(value).lower()
    )


PRIMARY_CARE_AMENITIES = {"clinic", "doctors", "health_post"}
# healthcare=* values that provide first-contact care; pharmacies, dentists,
# laboratories etc. are not counted as care destinations (A-Q06).
PRIMARY_CARE_HEALTHCARE = {"clinic", "doctor", "centre", "health_post"}


def destination_tier(props: Mapping) -> str:
    """Classify a destination's OSM tags; precedence hospital > primary_care > town.

    hospital: amenity=hospital or healthcare=hospital. primary_care:
    amenity=clinic/doctors/health_post or healthcare=clinic/doctor/centre/health_post.
    healthcare_other: any other healthcare=* value, including "no" (not a destination).
    town: place=town/city. Anything else is "other".
    """
    amenity = _tag(props, "amenity")
    healthcare = _tag(props, "healthcare")
    if "hospital" in {amenity, healthcare}:
        return "hospital"
    if amenity in PRIMARY_CARE_AMENITIES or healthcare in PRIMARY_CARE_HEALTHCARE:
        return "primary_care"
    if _tag(props, "place") in {"town", "city"}:
        return "town"
    if healthcare:  # includes healthcare=no: fetched by healthcare=*, not a destination
        return "healthcare_other"
    return "other"


def tag_destination_tiers(destinations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Copy with destination_tier; place and name always exist (null if absent)."""
    result = destinations.copy()
    for key in ("place", "name"):
        if key not in result.columns:
            result[key] = None
    result["destination_tier"] = [
        destination_tier(row) for _, row in result.iterrows()
    ]
    return result


def _xy_key(coordinate, signature):
    """Geometry-only node key; 1 micrometre rounding joins crossing points that two
    lines interpolate separately and that differ only in the last float bit."""
    return ("xy", round(coordinate[0], 6), round(coordinate[1], 6), *signature)


def build_graph(features, mode="vehicle"):
    """Build a directed graph; exact OSM node IDs take precedence over geometry."""
    _metric(features)
    if mode not in {"vehicle", "pedestrian"}:
        raise ValueError("mode must be vehicle or pedestrian")
    graph = nx.MultiDiGraph(crs=str(features.crs), mode=mode)
    roads = []
    for index, row in features.iterrows():
        highway = _tag(row, "highway")
        permission = _tag(row, "access")
        keys = (
            ("vehicle", "motor_vehicle", "motorcar") if mode == "vehicle" else ("foot",)
        )
        for key in keys:
            permission = _tag(row, key, permission)
        if highway in {"construction", "proposed", "abandoned", "razed"}:
            continue
        if permission in {"no", "private"} or (
            mode == "vehicle"
            and highway
            in {"footway", "path", "steps", "pedestrian", "cycleway", "bridleway"}
        ):
            continue
        if (
            mode == "pedestrian"
            and highway in {"motorway", "motorway_link"}
            and permission != "yes"
        ):
            continue
        geometry = row.geometry
        if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
            continue
        ids = row.get("node_ids")
        ids = (
            ids
            if isinstance(ids, (list, tuple)) and len(ids) == len(geometry.coords)
            else None
        )
        signature = (
            _tag(row, "layer", "0"),
            _tag(row, "bridge", "no"),
            _tag(row, "tunnel", "no"),
        )
        roads.append(
            (
                row,
                geometry,
                ids,
                signature,
                str(row.get("link_id", row.get("osm_id", row.get("id", index)))),
            )
        )
    endpoint_groups = {}
    transitions = nx.Graph()
    for _, line, ids, signature, _ in roads:
        if ids:
            continue
        for coordinate in (line.coords[0], line.coords[-1]):
            key = _xy_key(coordinate, signature)
            transitions.add_node(key)
            others = endpoint_groups.setdefault(tuple(coordinate[:2]), [])
            for other_key, other_signature in others:
                same_layer = signature[0] == other_signature[0]
                structure_transition = signature[1:] != other_signature[1:] and (
                    signature[1:] == ("no", "no") or other_signature[1:] == ("no", "no")
                )
                if same_layer or structure_transition:
                    transitions.add_edge(key, other_key)
            others.append((key, signature))
    endpoint_nodes = {
        node: min(component)
        for component in nx.connected_components(transitions)
        for node in component
    }
    positions = [
        np.r_[
            0,
            np.cumsum(
                np.linalg.norm(np.diff(np.asarray(line.coords)[:, :2], axis=0), axis=1)
            ),
        ]
        for _, line, _, _, _ in roads
    ]
    for distances, road in zip(positions, roads):
        distances[-1] = road[1].length
    cuts = [set(distances) for distances in positions]
    # OSM topology is authoritative; infer crossings only for geometry-only inputs.
    tree = STRtree([road[1] for road in roads])
    for i, (_, line, ids, signature, _) in enumerate(roads):
        if ids:
            continue
        for j in tree.query(line):
            if j >= i:
                continue
            _, other, other_ids, other_signature, _ = roads[j]
            if other_ids or signature != other_signature or not line.intersects(other):
                continue
            intersection = line.intersection(other)
            points = (
                [intersection]
                if intersection.geom_type == "Point"
                else list(intersection.geoms)
                if intersection.geom_type == "MultiPoint"
                else []
            )
            for point in points:
                cuts[i].add(line.project(point))
                cuts[j].add(other.project(point))
    for i, (row, line, ids, signature, link_id) in enumerate(roads):
        distances = sorted(cuts[i])
        original = {float(distance): k for k, distance in enumerate(positions[i])}
        for start, end in zip(distances, distances[1:]):
            if end <= start:
                continue
            segment = substring(line, start, end)
            nodes = []
            for distance, coordinate in (
                (start, segment.coords[0]),
                (end, segment.coords[-1]),
            ):
                node = (
                    ("osm", str(ids[original[distance]]))
                    if ids
                    else _xy_key(coordinate, signature)
                )
                if not ids:
                    node = endpoint_nodes.get(node, node)
                graph.add_node(node, x=coordinate[0], y=coordinate[1])
                nodes.append(node)
            attrs = dict(
                link_id=link_id,
                geometry=segment,
                length=segment.length,
                status=_tag(row, "operational_status", "unknown"),
            )
            oneway = _tag(row, "oneway" if mode == "vehicle" else "oneway:foot")
            if (
                mode == "vehicle"
                and not oneway
                and _tag(row, "junction") == "roundabout"
            ):
                oneway = "yes"
            if oneway != "-1":
                graph.add_edge(*nodes, **attrs)
            if oneway not in {"yes", "true", "1"}:
                graph.add_edge(*reversed(nodes), **attrs)
    return graph




NODE_TOLERANCE_M = 1e-8
SCENARIOS = ("1", "2", "3", "4")
SCENARIO_ASSUMPTIONS = {
    "1": "Hypothetical: links with an independently sourced confirmed closure are removed.",
    "2": "Hypothetical, primary: scenario 1 plus links with observed change (SAR water or disturbance inside valid coverage) or likely impact are removed.",
    "3": "Hypothetical, high confidence: scenario 1 plus radar-optical corroborated (likely impact) links are removed; a subset of scenario 2.",
    "4": "Hypothetical worst case: scenario 2 plus possible-impact, unobservable and operationally unknown links are removed.",
}
TARGETS = ("town_hospital", "primary_care")
ROUTE_COLUMNS = (
    "settlement_id",
    "mode",
    "scenario",
    "target",
    "destination_id",
    "destination_tier",
    "cost_m",
    "status",
    "link_ids",
    "geometry",
)


def _point(graph, node):
    return Point(graph.nodes[node]["x"], graph.nodes[node]["y"])


def snap_point(graph, point, tolerance):
    """Attach a point to its nearest graph edge within tolerance metres (stable API).

    Returns an end node of that edge when the orthogonal projection lies within
    1e-8 m of it. Otherwise every directed copy of the edge is split at the
    projection (one-way rules preserved) and the new node is returned. Returns
    None when no edge lies within tolerance. The graph is modified in place.
    On an exact distance tie (e.g. duplicate overlapping ways) the first edge in
    graph order is used, as before (P03 audit L2).
    """
    return _snap_points(graph, [point], tolerance)[0]


def _snap(point, graph, tolerance):
    """Deprecated alias of snap_point with the old argument order (flood-copilot uses it)."""
    return snap_point(graph, point, tolerance)


def _snap_points(graph, points, tolerance):
    """snap_point for many points: one STRtree lookup; each touched edge is split once."""
    edges = list(graph.edges(keys=True, data=True))
    if not edges or not points:
        return [None] * len(points)
    (point_index, edge_index), distances = STRtree(
        [data["geometry"] for *_, data in edges]
    ).query_nearest(points, all_matches=True, return_distance=True)
    nearest = {}
    for i, j, distance in zip(point_index.tolist(), edge_index.tolist(), distances.tolist()):
        # Equidistant edges: the first in graph edge order, as a linear scan picks.
        if i not in nearest or j < nearest[i][0]:
            nearest[i] = (j, distance)
    result = [None] * len(points)
    pending = {}  # one physical segment (both directed copies) -> its projections
    for i, (j, distance) in sorted(nearest.items()):
        if distance > tolerance:
            continue
        u, v, _, data = edges[j]
        line = data["geometry"]
        position = line.project(points[i])
        projection = line.interpolate(position)
        for node in (u, v):
            if projection.distance(_point(graph, node)) < NODE_TOLERANCE_M:
                result[i] = node
                break
        else:
            key = (frozenset((u, v)), data["link_id"])
            pending.setdefault(key, (u, v, []))[2].append((position, projection, i))
    for (_, link), (u, v, items) in pending.items():
        nodes = []
        for _, projection, i in sorted(items, key=lambda item: item[0]):
            if not nodes or projection.distance(nodes[-1][0]) >= NODE_TOLERANCE_M:
                node = ("snap", projection.x, projection.y, u, v)
                graph.add_node(node, x=projection.x, y=projection.y)
                nodes.append((projection, node))
            result[i] = nodes[-1][1]
        copies = [
            (a, b, k)
            for a, b in ([(u, v), (v, u)] if u != v else [(u, v)])
            for k, attrs in graph.get_edge_data(a, b, default={}).items()
            if attrs["link_id"] == link
        ]
        _split(graph, copies, nodes)
    return result


def _split(graph, copies, nodes):
    """Replace every directed copy of one segment by pieces through the new nodes."""
    for a, b, k in copies:
        attrs = graph.edges[a, b, k]
        segment = attrs["geometry"]
        forward = Point(segment.coords[0]).distance(_point(graph, a)) <= NODE_TOLERANCE_M
        cuts = sorted(
            ((segment.project(projection), node) for projection, node in nodes),
            key=lambda cut: cut[0],
        )
        chain = [a if forward else b, *(node for _, node in cuts), b if forward else a]
        positions = [0, *(position for position, _ in cuts), segment.length]
        graph.remove_edge(a, b, k)
        for left, right, start, end in zip(chain, chain[1:], positions, positions[1:]):
            part = substring(segment, start, end)
            edge = (left, right) if forward else (right, left)
            graph.add_edge(*edge, **{**attrs, "geometry": part, "length": part.length})


def _evidence(impacts):
    """link key -> the impact row's operational_status/evidence_category, if present."""
    for column in ("link_id", "osm_id", "id"):
        if column in impacts.columns:
            keys = impacts[column].tolist()
            break
    else:
        keys = impacts.index.tolist()
    columns = [
        c for c in ("operational_status", "evidence_category") if c in impacts.columns
    ]
    values = zip(*(impacts[c].tolist() for c in columns))
    return {str(key): dict(zip(columns, row)) for key, row in zip(keys, values)}


def _removal_sets(graphs, impacts):
    """Link ids removed in each hypothetical scenario, and operationally unknown links."""
    evidence = _evidence(impacts)
    removed = {scenario: set() for scenario in SCENARIOS}
    unknown = set()
    pairs = {
        (data["link_id"], data.get("status", "unknown"))
        for graph in graphs
        for _, _, data in graph.edges(data=True)
    }
    for link, edge_status in pairs:
        row = evidence.get(link, {})
        status = row.get("operational_status", edge_status)
        category = row.get("evidence_category", "unobservable")
        if status == "unknown":
            unknown.add(link)
        if status == "confirmed_open":
            continue  # never removed
        if status == "confirmed_closed":
            removed["1"].add(link)
        if category in {"observed_change", "likely_impact"}:
            removed["2"].add(link)
        if category == "likely_impact":
            removed["3"].add(link)
        if status == "unknown" or category in {"possible_impact", "unobservable"}:
            removed["4"].add(link)
    removed["2"] |= removed["1"]
    removed["3"] |= removed["1"]
    removed["4"] |= removed["2"]
    return removed, unknown


def _arrays(graph, codes):
    nodes = list(graph)
    index = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges(keys=True, data=True))
    return dict(
        graph=graph,
        index=index,
        edges=edges,
        tail=np.array([index[u] for u, _, _, _ in edges], dtype=np.int64),
        head=np.array([index[v] for _, v, _, _ in edges], dtype=np.int64),
        length=np.array([data["length"] for *_, data in edges], dtype=float),
        link=np.array([codes[data["link_id"]] for *_, data in edges], dtype=np.int64),
        searches={},
    )


def _search(net, removed, scenario, target):
    """Reverse multi-source Dijkstra from the targets over links kept in the scenario.

    Gives every node's cost to its nearest target, that target and the next node
    on the path, in one pass (cached per scenario and target set).
    """
    key = (scenario, target)
    if key not in net["searches"]:
        targets = net["targets"][target]
        if not targets:
            net["searches"][key] = None
            return None
        n = len(net["index"])
        keep = ~removed[scenario][net["link"]]
        rows, cols = net["head"][keep], net["tail"][keep]
        weights = net["length"][keep]
        order = np.lexsort((weights, cols, rows))
        rows, cols, weights = rows[order], cols[order], weights[order]
        first = np.ones(len(rows), bool)  # cheapest of parallel edges
        first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        matrix = csr_matrix(
            (weights[first], (rows[first], cols[first])), shape=(n, n)
        )  # explicit zeros stay zero-length edges
        cost, step, origin = dijkstra(
            matrix,
            directed=True,
            indices=sorted(targets),
            min_only=True,
            return_predecessors=True,
        )
        net["searches"][key] = dict(
            cost=cost,
            step=step,
            origin=origin,
            keys=rows[first] * n + cols[first],
            edges=np.flatnonzero(keep)[order][first],
        )
    return net["searches"][key]


def _network(
    settlements, destinations, vehicle_graph, pedestrian_graph, impacts, snap_tolerance
):
    """Validated inputs, removal sets and both graphs snapped in one batch each."""
    _metric(settlements)
    if snap_tolerance < 0:
        raise ValueError("snap_tolerance must be nonnegative")
    for frame in (destinations, impacts):
        if frame.crs != settlements.crs:
            raise ValueError("All vector CRSs must match")
    for graph in (vehicle_graph, pedestrian_graph):
        if CRS(graph.graph["crs"]) != CRS(settlements.crs):
            raise ValueError("Graph CRS must match settlements")
    tiers = (
        list(destinations["destination_tier"])
        if "destination_tier" in destinations.columns
        else [destination_tier(row) for _, row in destinations.iterrows()]
    )
    names = [
        str(name)
        for name in (
            destinations["id"] if "id" in destinations.columns else destinations.index
        )
    ]
    graphs = (vehicle_graph, pedestrian_graph)
    removed, unknown = _removal_sets(graphs, impacts)
    link_ids = sorted(
        {link for graph in graphs for _, _, link in graph.edges(data="link_id")}
    )
    codes = {link: i for i, link in enumerate(link_ids)}

    def mask(links):
        result = np.zeros(len(link_ids), bool)
        result[np.fromiter((codes[link] for link in links), np.int64)] = True
        return result

    places = [g.representative_point() for g in settlements.geometry]
    # Primary care is its own target set; other healthcare=* values are no destination.
    usable = [i for i, tier in enumerate(tiers) if tier != "healthcare_other"]
    stops = [destinations.geometry.iloc[i].representative_point() for i in usable]
    result = dict(
        link_ids=link_ids,
        removed={
            "baseline": mask(()),
            **{scenario: mask(removed[scenario]) for scenario in SCENARIOS},
        },
        unknown=mask(unknown),
    )
    for mode, graph in (("vehicle", vehicle_graph), ("pedestrian", pedestrian_graph)):
        graph = graph.copy()
        snapped = _snap_points(graph, places + stops, snap_tolerance)
        net = _arrays(graph, codes)
        net["sources"] = [
            None if node is None else net["index"][node]
            for node in snapped[: len(places)]
        ]
        net["targets"] = {target: {} for target in TARGETS}
        for i, node in zip(usable, snapped[len(places) :]):
            if node is None:
                continue
            group = net["targets"][
                "primary_care" if tiers[i] == "primary_care" else "town_hospital"
            ]
            current = group.get(net["index"][node])
            if current is None or names[i] < current[0]:  # co-located: smallest id
                group[net["index"][node]] = (names[i], tiers[i])
        result[mode] = net
    return result


def _nearest(network, mode, scenario, target, number):
    """(destination id, tier, network cost in metres) or None when not reachable."""
    net = network[mode]
    source = net["sources"][number]
    search = _search(net, network["removed"], scenario, target)
    if search is None or source is None or not np.isfinite(search["cost"][source]):
        return None
    name, tier = net["targets"][target][int(search["origin"][source])]
    return name, tier, float(search["cost"][source])


def _path_edges(net, search, source):
    """Directed edges, in order, from source to its nearest target."""
    if "step_list" not in search:
        search["step_list"] = search["step"].tolist()
    step = search["step_list"]
    nodes = [source]
    while step[nodes[-1]] >= 0:
        nodes.append(step[nodes[-1]])
    n = len(net["index"])
    keys = np.array(nodes[1:], np.int64) * n + np.array(nodes[:-1], np.int64)
    positions = search["edges"][np.searchsorted(search["keys"], keys)]
    return [net["edges"][position] for position in positions.tolist()]


def _reachable_links(net, links):
    """Per source: links whose edge starts at a node reachable at baseline.

    Sources in one strongly connected component share the same set, computed once.
    """
    n = len(net["index"])
    forward = csr_matrix(
        (np.ones(len(net["tail"])), (net["tail"], net["head"])), shape=(n, n)
    )
    _, labels = connected_components(forward, directed=True, connection="strong")
    masks, keys = {}, []
    for source in net["sources"]:
        if source is None:
            keys.append(None)
            continue
        label = int(labels[source])
        if label not in masks:
            reached = np.zeros(n, bool)
            reached[breadth_first_order(forward, source, return_predecessors=False)] = True
            mask = np.zeros(links, bool)
            mask[net["link"][reached[net["tail"]]]] = True
            masks[label] = mask
        keys.append(label)
    masks[None] = np.zeros(links, bool)
    return keys, masks


SETTLEMENT_COLUMNS = (
    "baseline_accessible",
    "baseline_incomplete",
    "destination_coverage",
    "scenario1_accessible",
    "scenario2_accessible",
    "scenario3_accessible",
    "scenario4_accessible",
    "potentially_isolated",
    "isolation_confidence",
    "cut_in_scenarios",
    "depends_only_on_unknown",
    "pedestrian_accessible",
    "depends_on_unknown_links",
    "affected_connections",
    "candidate_link_ids",
    "candidate_link_count_worst_case",
    "scenario1_cut_link_ids",
    "scenario2_cut_link_ids",
    "scenario3_cut_link_ids",
    "scenario4_cut_link_count",
    "primary_care_baseline_accessible",
    "primary_care_scenario1_accessible",
    "primary_care_scenario2_accessible",
    "primary_care_scenario3_accessible",
    "primary_care_scenario4_accessible",
    "primary_care_coverage",
    "nearest_destination_id",
    "nearest_destination_tier",
    "nearest_destination_cost_m",
    "scenario2_nearest_destination_id",
    "scenario2_nearest_destination_cost_m",
    "scenario2_detour_m",
    "nearest_primary_care_id",
    "nearest_primary_care_cost_m",
    "scenario2_nearest_primary_care_id",
    "scenario2_nearest_primary_care_cost_m",
)


def analyze(
    settlements,
    destinations,
    vehicle_graph,
    pedestrian_graph,
    impacts,
    snap_tolerance=200,
):
    """Accessibility is mapped reachability, never proof of operational road access.

    Settlements, destinations and impacts share one metric CRS with the graphs.
    Scenarios 1-4 are hypothetical link removals (SCENARIO_ASSUMPTIONS);
    potentially_isolated is accessible at baseline but not in scenario 2.
    Nearest-destination costs are network metres. routes() returns the paths.
    """
    network = _network(
        settlements, destinations, vehicle_graph, pedestrian_graph, impacts, snap_tolerance
    )
    vehicle = network["vehicle"]
    link_ids, removed = network["link_ids"], network["removed"]
    labels, masks = _reachable_links(vehicle, len(link_ids))
    has_targets = bool(vehicle["targets"]["town_hospital"])
    has_care = bool(vehicle["targets"]["primary_care"])
    cache = {}

    def removed_ids(label, scenario):
        if (label, scenario) not in cache:
            selected = np.flatnonzero(masks[label] & removed[scenario])
            cache[label, scenario] = [link_ids[i] for i in selected]
        return list(cache[label, scenario])

    records = []
    for number in range(len(settlements)):
        label = labels[number]
        relevant = masks[label]
        town = {
            scenario: _nearest(network, "vehicle", scenario, "town_hospital", number)
            for scenario in ("baseline", *SCENARIOS)
        }
        care = {
            scenario: _nearest(network, "vehicle", scenario, "primary_care", number)
            for scenario in ("baseline", *SCENARIOS)
        }
        baseline = town["baseline"] is not None
        accessible = {scenario: town[scenario] is not None for scenario in SCENARIOS}
        cut = [int(s) for s in SCENARIOS if baseline and not accessible[s]]
        record = dict(
            baseline_accessible=baseline,
            baseline_incomplete=not baseline or not has_targets,
            destination_coverage="mapped_destinations_available"
            if has_targets
            else "missing_or_unsnapped_destinations",
            **{f"scenario{s}_accessible": accessible[s] for s in SCENARIOS},
            potentially_isolated=baseline and not accessible["2"],
            # S1 is a subset of S3, so a cut by confirmed closures alone is also
            # "corroborated" (audit L3; no confirmed statuses exist in current runs).
            isolation_confidence="corroborated"
            if 3 in cut
            else "observed"
            if 2 in cut
            else None,
            cut_in_scenarios=cut,
            depends_only_on_unknown=baseline
            and accessible["2"]
            and not accessible["4"],
            pedestrian_accessible=_nearest(
                network, "pedestrian", "2", "town_hospital", number
            )
            is not None,
            depends_on_unknown_links=bool((relevant & network["unknown"]).any()),
            affected_connections=int((relevant & removed["2"]).sum()),
            # Links worth inspecting: reachable links in the primary (observed) removal
            # set. Under scenario 4 nearly every link is "unknown", so only counts are
            # kept for it (2.0; a full list was 10M ids at Melamchi scale, HR-X-007).
            candidate_link_ids=removed_ids(label, "2"),
            candidate_link_count_worst_case=len(removed_ids(label, "4")),
            **{
                f"scenario{s}_cut_link_ids": removed_ids(label, s)
                if int(s) in cut
                else []
                for s in ("1", "2", "3")
            },
            scenario4_cut_link_count=len(removed_ids(label, "4")) if 4 in cut else 0,
            # Unknown (None), not False, when no primary care is mapped or snapped.
            primary_care_baseline_accessible=care["baseline"] is not None
            if has_care
            else None,
            **{
                f"primary_care_scenario{s}_accessible": care[s] is not None
                if has_care
                else None
                for s in SCENARIOS
            },
            primary_care_coverage="mapped_primary_care_available"
            if has_care
            else "missing_or_unsnapped_primary_care",
        )
        record.update(
            nearest_destination_id=_field(town["baseline"], 0),
            nearest_destination_tier=_field(town["baseline"], 1),
            nearest_destination_cost_m=_field(town["baseline"], 2),
            scenario2_nearest_destination_id=_field(town["2"], 0),
            scenario2_nearest_destination_cost_m=_field(town["2"], 2),
            scenario2_detour_m=town["2"][2] - town["baseline"][2]
            if town["baseline"] and town["2"]
            else None,
            nearest_primary_care_id=_field(care["baseline"], 0),
            nearest_primary_care_cost_m=_field(care["baseline"], 2),
            scenario2_nearest_primary_care_id=_field(care["2"], 0),
            scenario2_nearest_primary_care_cost_m=_field(care["2"], 2),
        )
        records.append(record)
    result = settlements.copy()
    for key in SETTLEMENT_COLUMNS:
        values = [record[key] for record in records]
        # Nullable numbers stay None (unknown), not NaN, in memory.
        result[key] = (
            pd.Series(values, index=result.index, dtype=object)
            if key.endswith("_m")
            else values
        )
    result.attrs["scenario_assumptions"] = dict(SCENARIO_ASSUMPTIONS)
    result.attrs["limitations"] = (
        "Mapped reachability only; destination and network completeness are not assured. Cut candidates include removed reachable links, not a proven minimal cut."
        " Scenarios are hypothetical link removals; links with a confirmed-open status are never removed."
        " Nearest-destination costs are shortest mapped network distances in metres, not travel times or usable routes."
        " Primary-care facilities (clinics, doctors, health posts, healthcare=clinic/doctor/centre) are a separate destination tier, excluded from baseline/scenario accessibility and potentially_isolated; mapped reachability is not proof of service availability."
    )
    result.attrs["destination_tiers"] = {
        "accessibility": "towns, cities, hospitals (and untagged destinations)",
        "primary_care": "amenity=clinic/doctors/health_post or healthcare=clinic/doctor/centre/health_post",
        "excluded": "other healthcare=* values (pharmacy, dentist, laboratory, ...)",
    }
    return result


def routes(
    settlements,
    destinations,
    vehicle_graph,
    pedestrian_graph,
    impacts,
    snap_tolerance=200,
):
    """Shortest mapped network path from each settlement to its nearest target.

    One row per settlement, mode (vehicle, pedestrian), scenario (baseline,
    1-4) and target set (town_hospital, primary_care), in the settlements' CRS.
    Snapping and removal sets are those of analyze() on the same inputs, so
    status "found" agrees with its accessibility flags. status is "found",
    "none" (no mapped path) or "unassessable" (settlement unsnapped, or no
    snapped target). link_ids lists links in path order; geometry is None when
    no path is found or the settlement snaps onto the target (cost 0). A path
    is mapped reachability under a hypothetical scenario, never a safe route.
    """
    network = _network(
        settlements, destinations, vehicle_graph, pedestrian_graph, impacts, snap_tolerance
    )
    ids = settlements["id"] if "id" in settlements.columns else settlements.index
    rows = []
    for number, settlement_id in enumerate(ids):
        for mode in ("vehicle", "pedestrian"):
            net = network[mode]
            source = net["sources"][number]
            for scenario in ("baseline", *SCENARIOS):
                for target in TARGETS:
                    row = dict(
                        settlement_id=str(settlement_id),
                        mode=mode,
                        scenario=scenario,
                        target=target,
                        destination_id=None,
                        destination_tier=None,
                        cost_m=None,
                        status="unassessable",
                        link_ids=[],
                        geometry=None,
                    )
                    rows.append(row)
                    if source is None or not net["targets"][target]:
                        continue
                    found = _nearest(network, mode, scenario, target, number)
                    if found is None:
                        row["status"] = "none"
                        continue
                    edges = _path_edges(
                        net, _search(net, network["removed"], scenario, target), source
                    )
                    row.update(
                        status="found",
                        destination_id=found[0],
                        destination_tier=found[1],
                        cost_m=found[2],
                        link_ids=_path_links(edges),
                        geometry=_line(net["graph"], edges),
                    )
    frame = gpd.GeoDataFrame(
        {
            column: pd.Series([row[column] for row in rows], dtype=object)
            if column == "cost_m"
            else [row[column] for row in rows]
            for column in ROUTE_COLUMNS
        },
        geometry="geometry",
        crs=settlements.crs,
    )
    frame.attrs["scenario_assumptions"] = dict(SCENARIO_ASSUMPTIONS)
    frame.attrs["limitations"] = (
        "Shortest mapped network paths in metres under hypothetical link removals; "
        "not travel times, and never safe, open or usable routes."
    )
    return frame


def _field(found, index):
    return found[index] if found else None


def _path_links(edges):
    """Link ids along a path; consecutive pieces of one link are listed once."""
    links = []
    for *_, data in edges:
        if not links or links[-1] != data["link_id"]:
            links.append(data["link_id"])
    return links


def _line(graph, edges):
    coordinates = []
    for u, _, _, data in edges:
        part = list(data["geometry"].coords)
        start = _point(graph, u)
        if Point(part[-1]).distance(start) < Point(part[0]).distance(start):
            part.reverse()
        coordinates.extend(part if not coordinates else part[1:])
    return LineString(coordinates) if coordinates else None
