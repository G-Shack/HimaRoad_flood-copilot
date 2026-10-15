"""Reviewed hazard-v2 schema3.0 invariants; no reference/model evaluation reads."""

import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from shapely.geometry import shape

from backend.app.settings import Settings
from backend.app.upstream.paths import RunContractError, authorised_path
from backend.app.upstream.v1 import time, verify_osm_proof
from backend.app.upstream.v1 import validate_metadata as provenance

SCENARIOS = ("1", "2", "3", "4")
EXTRA_RASTERS = (
    "dem",
    "layover",
    "radar_shadow",
    "possible_debris",
    "permanent_water",
    "ephemeral_water",
    "change_zscore",
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RunContractError("contract 3.0: " + reason)


def finite(value, *, minimum=0, maximum=None) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(value)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def metre_crs(value) -> bool:
    crs = CRS.from_user_input(value)
    return bool(
        crs.is_projected
        and len(crs.axis_info) >= 2
        and all(
            axis.unit_conversion_factor == 1 and axis.unit_name.lower() in {"metre", "meter"}
            for axis in crs.axis_info[:2]
        )
    )


def vector(root: Path, name: str) -> list[dict]:
    obj = json.loads((root / (name + ".geojson")).read_text())
    require(
        obj.get("type") == "FeatureCollection" and isinstance(obj.get("features"), list),
        "vector structure",
    )
    return obj["features"]


def numeric_coordinates(value) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if isinstance(value[0], (list, tuple)):
        return all(numeric_coordinates(item) for item in value)
    return all(type(item) in {int, float} and math.isfinite(item) for item in value)


def geometry(value, *, nullable=False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, dict):
        raise RunContractError("contract 3.0: geometry must be an object")
    require(numeric_coordinates(value.get("coordinates")), "non-numeric coordinates")
    geom = shape(value)
    require(not geom.is_empty and geom.is_valid, "invalid geometry")
    x0, y0, x1, y1 = geom.bounds
    require(-180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90, "non-WGS84 geometry")


def validate_metadata(
    root: Path, report: dict, manifest: dict, settings: Settings
) -> tuple[dict, list[dict]]:
    metadata = provenance(report, manifest)
    if report["status"] == "incomplete":
        return metadata, []
    verify_osm_proof(root, manifest, settings)
    require(
        report.get("primary_scenario") == "2" and set(report["scenarios"]) == set(SCENARIOS),
        "scenario meanings missing",
    )
    require(report.get("detector") in {"baseline", "unet"}, "unreviewed detector")
    for key in ("hazard_version", "hazard", "sar_geometry", "hazard_quality_flags"):
        require(
            key in report and key in manifest and report[key] == manifest[key], "conflicting " + key
        )
    require(report["hazard_version"] == "hazard-v2", "only hazard-v2 is reviewed")
    require(
        set(report["summary"])
        == {
            "potentially_road_cut",
            "corroborated_road_cut",
            "access_uncertain",
            "depends_only_on_unknown",
            "settlements_total",
            "settlements_from_building_clusters",
            "trail_access_possible",
            "primary_care_potentially_cut",
            "potentially_exposed_buildings",
            "suspected_affected_links",
            "valid_post_observation_percent",
            "valid_change_observation_percent",
        },
        "unreviewed summary fields",
    )
    hazard = report["hazard"]
    require(isinstance(hazard, dict), "hazard provenance missing")
    require(hazard.get("hazard_version") == "hazard-v2", "hazard version differs")
    fusion = hazard.get("fusion", {})
    tracks = fusion.get("tracks")
    require(isinstance(tracks, list) and 1 <= len(tracks) <= 3, "fusion track inventory")
    pairs = [(manifest["pre"], manifest["post"])]
    extras = manifest.get("tracks", [])
    require(isinstance(extras, list), "extra track inventory")
    pairs.extend((track["pre"], track["post"]) for track in extras)
    require(len(tracks) == len(pairs), "fusion/acquisition track count differs")
    event = metadata["event_time"] or metadata["event_date_boundary"]
    post_min = (
        event
        if metadata["event_time"] or "T" in report["event_date"]
        else event + timedelta(days=1)
    )
    cutoff = time(manifest["cutoff"]) if manifest.get("cutoff") else None
    acquisitions = []
    track_posts = {}
    for (pre, post), fused in zip(pairs, tracks, strict=True):
        require(isinstance(pre, list) and bool(pre), "pre scene inventory")
        require(
            time(fused["post_timestamp"]) == time(post["timestamp"]), "fusion post time differs"
        )
        track = fused.get("track")
        require(isinstance(track, str) and track not in track_posts, "duplicate fusion track")
        track_posts[track] = time(post["timestamp"])
        require(
            time(post["timestamp"]) >= post_min
            and (cutoff is None or time(post["timestamp"]) <= cutoff),
            "post event/cutoff bounds",
        )
        for scene in pre:
            require(time(scene["timestamp"]) < event, "pre event bounds")
            for key in ("relative_orbit", "pass", "instrument_mode", "polarization", "units"):
                require(key in scene and scene[key] == post.get(key), "cross-track pairing")
        for scene in (*pre, post):
            require(isinstance(scene.get("source"), str), "source provenance missing")
            if not report["synthetic"]:
                require(manifest.get("provider") == "cdse", "real provider is unreviewed")
                require(
                    bool(scene.get("scene_id")) and bool(scene.get("stac_item_ids")),
                    "real scene IDs missing",
                )
                processing = scene.get("processing", {})
                require(
                    "CDSE" in scene["source"]
                    and processing.get("backCoeff") == "GAMMA0_TERRAIN"
                    and processing.get("demInstance") == "COPERNICUS_30",
                    "real SAR processing differs",
                )
                masks = scene.get("masks", {})
                require(
                    masks.get("bands") == ["dataMask", "shadowMask", "localIncidenceAngle"],
                    "provider masks missing",
                )
                path = authorised_path(root / masks["path"], settings, root=root)
                with rasterio.open(path) as data:
                    require(data.count == 3 and metre_crs(data.crs), "provider mask bands/CRS")
                    for band in (1, 2):
                        values = data.read(band, masked=True).compressed()
                        require(bool(np.isin(values, [0, 1]).all()), "invalid mask code")
            acquisitions.append(scene)
    for key in ("optical_pre", "optical_post"):
        scene = manifest.get(key)
        if not scene:
            continue
        timestamp = time(scene["timestamp"])
        require(
            timestamp < event if key == "optical_pre" else timestamp >= post_min,
            "optical event bounds",
        )
        require(
            key == "optical_pre" or cutoff is None or timestamp <= cutoff, "optical cutoff bounds"
        )
        if not report["synthetic"]:
            require(
                bool(scene.get("scene_id")) and "CDSE" in scene.get("source", ""),
                "optical source identity",
            )
        acquisitions.append(scene)
    optical = hazard.get("optical", {})
    require(optical.get("clear_classes") == [4, 5, 6], "optical clear classes differ")
    if optical.get("gap_days") is not None:
        require(finite(optical["gap_days"]), "optical gap invalid")
        require(
            optical.get("seasonal_prone")
            == (optical["gap_days"] > hazard["settings"]["optical_max_gap_days"]),
            "optical gap policy differs",
        )
    require(type(hazard.get("mmu_pixels")) is int and hazard["mmu_pixels"] >= 1, "MMU unavailable")
    if not report["synthetic"]:
        require(
            "COPERNICUS" in manifest.get("dem", {}).get("source", "").upper(),
            "DEM source is unreviewed",
        )
    for key in (
        "layover_percent",
        "geometric_shadow_percent",
        "provider_shadow_percent",
        "masked_percent",
    ):
        require(
            finite(report["sar_geometry"].get(key), maximum=100),
            "shadow/layover coverage unavailable",
        )
    require(
        manifest.get("valid_observation_coverage")
        == report["summary"]["valid_change_observation_percent"] / 100,
        "coverage differs",
    )
    hand = report["hazard_quality_flags"].get("hand_soft_flag")
    if hand:
        require(
            hand.get("policy") == "candidates above max_hand_m are kept unchanged and only counted",
            "HAND must remain count-only",
        )
    # Fused counts refer to output masks; per-track counters stay separate.
    for layer in (
        "flood_extent",
        "disturbance",
        "possible_debris",
        "permanent_water",
        "ephemeral_water",
    ):
        with rasterio.open(root / (layer + ".tif")) as raster:
            require(metre_crs(raster.crs), "hazard metric CRS axes")
            values = raster.read(1, masked=True)
            require(bool(np.isin(values.compressed(), [0, 1]).all()), "binary layer code")
            count = int(np.count_nonzero(values.filled(0) == 1))
            require(
                type(hazard["pixels"].get(layer)) is int and hazard["pixels"][layer] == count,
                "fused layer count differs: " + layer,
            )
    with rasterio.open(root / "disturbance.tif") as data:
        disturbance = data.read(1, masked=True).filled(0) == 1
    with rasterio.open(root / "possible_debris.tif") as data:
        require(
            not bool(((data.read(1, masked=True).filled(0) == 1) & ~disturbance).any()),
            "possible debris is not a disturbance subset",
        )
    # Each contributing feature keeps its earliest contributing SAR post instant.
    for layer in ("flood_extent", "disturbance", "possible_debris", "roads", "buildings"):
        for feature in vector(root, layer):
            props = feature["properties"]
            stamps, names = props.get("sar_post_timestamps"), props.get("sar_tracks")
            require(
                isinstance(stamps, list) and isinstance(names, list) and len(stamps) == len(names),
                "feature track provenance missing",
            )
            require(
                all(
                    name in track_posts and time(stamp) == track_posts[name]
                    for name, stamp in zip(names, stamps, strict=True)
                ),
                "feature contributing scene differs",
            )
            if stamps:
                require(
                    time(props["observation_timestamp"]) == min(map(time, stamps)),
                    "feature time is not earliest contributing post",
                )
            if layer in {"flood_extent", "disturbance", "possible_debris"}:
                require(
                    bool(stamps) and props.get("hazard_version") == "hazard-v2",
                    "hazard feature method/time missing",
                )
            if layer in {"roads", "buildings"}:
                if props["evidence_category"] in {"observed_change", "likely_impact"}:
                    require(
                        props["flood_intersection"] or props["disturbance_intersection"],
                        "optical-only evidence upgraded",
                    )
                if (
                    optical.get("seasonal_prone")
                    and props["evidence_category"] == "possible_impact"
                ):
                    require(
                        props["flood_intersection"] or props["disturbance_intersection"],
                        "long-gap optical-only evidence not suppressed",
                    )
    validate_routes(root)
    validate_flow(root, manifest, settings)
    return metadata, acquisitions


def validate_routes(root: Path) -> None:
    settlements = {f["properties"]["id"]: f["properties"] for f in vector(root, "settlements")}
    destinations = {f["properties"]["id"] for f in vector(root, "destinations")}
    roads = {f["properties"]["link_id"] for f in vector(root, "roads")}
    seen = set()
    route_rows = {}
    for feature in vector(root, "routes"):
        props = feature["properties"]
        sid, mode, scenario, target = (
            props.get(k) for k in ("settlement_id", "mode", "scenario", "target")
        )
        require(
            sid in settlements
            and mode in {"vehicle", "pedestrian"}
            and scenario in {"baseline", *SCENARIOS}
            and target in {"town_hospital", "primary_care"},
            "route scope invalid",
        )
        key = sid, mode, scenario, target
        require(key not in seen, "duplicate route")
        seen.add(key)
        route_rows[key] = props
        status = props.get("status")
        require(status in {"found", "none", "unassessable"}, "route status invalid")
        links = props.get("link_ids")
        require(
            isinstance(links, list) and all(link in roads for link in links), "foreign route link"
        )
        if status == "found":
            require(
                props.get("destination_id") in destinations and finite(props.get("cost_m")),
                "found route target/cost invalid",
            )
            require(
                feature["geometry"] is not None or props["cost_m"] == 0,
                "nonzero route geometry missing",
            )
        else:
            require(
                props.get("cost_m") is None
                and props.get("destination_id") is None
                and not links
                and feature["geometry"] is None,
                "unknown/absent route filled with facts",
            )
        geometry(feature["geometry"], nullable=True)
        if mode == "vehicle":
            field = (
                "baseline_accessible"
                if scenario == "baseline"
                else f"scenario{scenario}_accessible"
            )
            if target == "primary_care":
                field = "primary_care_" + field
            expected = settlements[sid][field]
            if expected is not None:
                require(expected == (status == "found"), "route accessibility mismatch")
    expected = {
        (sid, mode, scenario, target)
        for sid in settlements
        for mode in ("vehicle", "pedestrian")
        for scenario in ("baseline", *SCENARIOS)
        for target in ("town_hospital", "primary_care")
    }
    require(seen == expected, "persisted route inventory incomplete")
    for sid, props in settlements.items():
        for scenario, target, id_field, cost_field in (
            ("baseline", "town_hospital", "nearest_destination_id", "nearest_destination_cost_m"),
            (
                "2",
                "town_hospital",
                "scenario2_nearest_destination_id",
                "scenario2_nearest_destination_cost_m",
            ),
            ("baseline", "primary_care", "nearest_primary_care_id", "nearest_primary_care_cost_m"),
            (
                "2",
                "primary_care",
                "scenario2_nearest_primary_care_id",
                "scenario2_nearest_primary_care_cost_m",
            ),
        ):
            row = route_rows[sid, "vehicle", scenario, target]
            require(
                props.get(id_field) == row["destination_id"]
                and props.get(cost_field) == row["cost_m"],
                "nearest target contradicts persisted route",
            )
        require(
            all(type(props.get(f"scenario{s}_accessible")) is bool for s in SCENARIOS),
            "scenario access unavailable",
        )
        for s in SCENARIOS:
            value = props.get(f"primary_care_scenario{s}_accessible")
            require(value is None or type(value) is bool, "primary-care access invalid")
            require(
                (value is None)
                == (props.get("primary_care_coverage") == "missing_or_unsnapped_primary_care"),
                "primary-care unknowns differ",
            )
        require(type(props.get("depends_only_on_unknown")) is bool, "unknown-link access absent")
        require(
            props["depends_only_on_unknown"]
            == (
                props["baseline_accessible"]
                and props["scenario2_accessible"]
                and not props["scenario4_accessible"]
            ),
            "scenario4 unknown semantics differ",
        )
        require(
            props.get("cut_in_scenarios")
            == [
                int(s)
                for s in SCENARIOS
                if props["baseline_accessible"] and not props[f"scenario{s}_accessible"]
            ],
            "cut scenario identity differs",
        )
        require(
            props["potentially_isolated"]
            == (props["baseline_accessible"] and not props["scenario2_accessible"]),
            "new isolation contradicts baseline",
        )
        for key in ("kind", "name_source", "cluster_id", "represents_cluster", "building_count"):
            require(key in props, "cluster/name provenance missing")
        for key in (
            "nearest_destination_id",
            "nearest_destination_cost_m",
            "scenario2_nearest_destination_id",
            "scenario2_nearest_destination_cost_m",
            "nearest_primary_care_id",
            "nearest_primary_care_cost_m",
            "scenario2_nearest_primary_care_id",
            "scenario2_nearest_primary_care_cost_m",
        ):
            require(key in props, "nearest target unavailable")
            if key.endswith("_m"):
                require(props[key] is None or finite(props[key]), "nearest cost invalid")


def validate_flow(root: Path, manifest: dict, settings: Settings) -> None:
    if not (root / "flowpaths.geojson").exists():
        return
    dem = manifest.get("dem", {})
    require(isinstance(dem.get("path"), str), "flow DEM provenance missing")
    with rasterio.open(authorised_path(root / dem["path"], settings, root=root)) as data:
        require(metre_crs(data.crs), "flow DEM metric axes")
    settlements = {f["properties"]["id"] for f in vector(root, "settlements")}
    seen = set()
    for feature in vector(root, "flowpaths"):
        props = feature["properties"]
        require(isinstance(props.get("id"), str) and props["id"] not in seen, "flow identity")
        seen.add(props["id"])
        geometry(feature["geometry"])
        require(props.get("kind") in {"flow_path", "corridor"}, "flow kind")
        require(finite(props.get("corridor_m")), "flow corridor units")
        require("not an inundation forecast" in props.get("label", ""), "flow limitation absent")
        if props["kind"] == "corridor":
            continue
        require(
            props.get("method")
            == "D8 steepest descent on priority-flood conditioned DEM (Barnes 2014)",
            "unreviewed flow method",
        )
        require(type(props.get("cells")) is int and props["cells"] >= 1, "flow cells missing")
        require(
            finite(props.get("length_m")) and finite(props.get("start_snapped_m")),
            "flow lengths invalid",
        )
        require(
            isinstance(props.get("dem_source"), str) and bool(props["dem_source"]),
            "flow DEM source missing",
        )
        rows = props.get("settlements")
        require(isinstance(rows, list), "flow settlements missing")
        last = -1
        for row in rows:
            require(
                row.get("settlement_id") in settlements
                and finite(row.get("along_m"))
                and finite(row.get("offset_m")),
                "foreign/invalid flow settlement",
            )
            require(
                last <= row["along_m"] <= props["length_m"] + 1e-6
                and row["offset_m"] <= props["corridor_m"] + 1e-6,
                "flow ordering/corridor differs",
            )
            last = row["along_m"]
