"""Pinned, observed output vocabulary; no upstream implementation is copied."""

RASTERS = (
    "pre",
    "post",
    "valid",
    "aoi",
    "flood_extent",
    "disturbance",
    "unobservable",
    "optical_water",
    "optical_disturbance",
)
VECTORS = (
    "flood_extent",
    "disturbance",
    "unobservable",
    "optical_water",
    "optical_disturbance",
    "roads",
    "affected_infrastructure",
    "buildings",
    "settlements",
    "destinations",
)
SUMMARY_COUNTS = (
    "potentially_road_cut",
    "access_uncertain",
    "trail_access_possible",
    "potentially_exposed_buildings",
    "suspected_affected_links",
)
SUMMARY_PERCENTAGES = ("valid_post_observation_percent", "valid_change_observation_percent")
EVIDENCE = {
    "observed_change",
    "likely_impact",
    "possible_impact",
    "unobservable",
    "no_detected_change",
}
SETTLEMENT_CATEGORIES = {
    "potentially_road_cut",
    "access_uncertain",
    "baseline_network_incomplete",
    "mapped_road_connection_remains",
}
ASSET_FIELDS = (
    "id",
    "evidence_category",
    "operational_status",
    "physical_condition",
    "valid_observation_fraction",
    "flood_intersection",
    "disturbance_intersection",
    "potentially_exposed",
    "observation_timestamp",
)
SETTLEMENT_FIELDS = (
    "id",
    "name",
    "baseline_accessible",
    "baseline_incomplete",
    "destination_coverage",
    "scenario1_accessible",
    "scenario2_accessible",
    "scenario3_accessible",
    "potentially_isolated",
    "pedestrian_accessible",
    "depends_on_unknown_links",
    "affected_connections",
    "candidate_link_ids",
    "scenario1_cut_link_ids",
    "scenario2_cut_link_ids",
    "scenario3_cut_link_ids",
    "category",
    "connectivity_scenario",
    "observation_timestamp",
    "source_dataset",
    "evidence_category",
    "potentially_exposed_buildings",
    "evidence_strength",
)


def vectors(schema_version: str | None) -> tuple[str, ...]:
    return (*VECTORS, "possible_debris") if schema_version == "3.0" else VECTORS


def rasters(schema_version: str | None) -> tuple[str, ...]:
    if schema_version == "3.0":
        from backend.app.upstream.v3 import EXTRA_RASTERS

        return (*RASTERS, *EXTRA_RASTERS)
    return RASTERS
