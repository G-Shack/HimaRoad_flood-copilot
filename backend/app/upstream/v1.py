"""Reviewed FC-C-007 + FC-C-008 metadata and temporal invariants."""

import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.upstream.paths import RunContractError

PROVENANCE = (
    "schema_version",
    "producer_commit",
    "producer_tree_dirty",
    "producer_commit_source",
    "package_version",
)
CARE_FIELDS = (
    "primary_care_baseline_accessible",
    *(f"primary_care_scenario{i}_accessible" for i in (1, 2, 3)),
)
TIERS = {"hospital", "primary_care", "town", "healthcare_other", "other"}


def time(value: Any) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timezone")
        return result.astimezone(UTC)
    except (ValueError, AttributeError) as exc:
        raise RunContractError("contract 1.0: expected timezone-aware timestamp") from exc


def validate_metadata(report: dict, manifest: dict) -> dict:
    for field in PROVENANCE:
        if (
            field not in report
            or field not in manifest
            or report[field] != manifest[field]
            or type(report[field]) is not type(manifest[field])
        ):
            raise RunContractError(f"contract 1.0 provenance: missing/conflicting {field}")
    producer = report["producer_commit"]
    if producer != "unknown" and (
        not isinstance(producer, str) or not re.fullmatch("[0-9a-f]{40}", producer)
    ):
        raise RunContractError("contract 1.0: invalid producer_commit")
    dirty = report["producer_tree_dirty"]
    origin = report["producer_commit_source"]
    if dirty is not None and type(dirty) is not bool:
        raise RunContractError("contract 1.0: dirty flag must be bool or null")
    if origin not in {"git", "sync_stamp", "unavailable"}:
        raise RunContractError("contract 1.0: unknown producer_commit_source")
    if (producer == "unknown") != (origin == "unavailable") or (
        producer == "unknown" and dirty is not None
    ):
        raise RunContractError("contract 1.0: inconsistent producer identity")
    if not isinstance(report["package_version"], str) or not report["package_version"]:
        raise RunContractError("contract 1.0: missing package_version")
    if not isinstance(manifest.get("limitations"), list) or any(
        not isinstance(s, str) for s in manifest["limitations"]
    ):
        raise RunContractError("contract 1.0: manifest limitations must be strings")
    for field in ("event_time", "temporal_offsets"):
        if field not in report:
            raise RunContractError(f"contract 1.0 report: missing {field}")
        if field in manifest and manifest[field] != report[field]:
            raise RunContractError(f"contract 1.0: conflicting {field}")
    event = report["event_date"]
    boundary = time(event) if "T" in event else time(event + "T00:00:00Z")
    event_time = time(report["event_time"]) if report["event_time"] is not None else None
    if event_time is not None and not boundary - timedelta(
        days=1
    ) < event_time < boundary + timedelta(days=1):
        raise RunContractError("contract 1.0: event_time outside allowed boundary interval")
    offsets = report["temporal_offsets"]
    if report["status"] == "complete":
        if not isinstance(offsets, dict) or offsets != manifest.get("temporal_offsets"):
            raise RunContractError("contract 1.0: complete temporal offsets missing/conflicting")
        expected_basis = (
            "hours after the supplied event-date timestamp (UTC)"
            if "T" in event
            else "hours after the supplied event-date boundary (00:00 UTC)"
        )
        if (
            offsets.get("basis") != expected_basis
            or time(offsets.get("event_date_boundary_utc")) != boundary
        ):
            raise RunContractError("contract 1.0: temporal offset basis mismatch")
        pre = [time(s["timestamp"]) for s in manifest["pre"]]
        post = time(manifest["post"]["timestamp"])
        expected = {
            "post_hours_after_event_date_boundary": (post - boundary).total_seconds() / 3600,
            "latest_pre_hours_before_event_date_boundary": (boundary - max(pre)).total_seconds()
            / 3600,
            "post_hours_after_event_time": (post - event_time).total_seconds() / 3600
            if event_time
            else None,
            "latest_pre_hours_before_event_time": (event_time - max(pre)).total_seconds() / 3600
            if event_time
            else None,
        }
        for key, value in expected.items():
            observed: Any = offsets.get(key)
            if value is None:
                if observed is not None:
                    raise RunContractError("contract 1.0: missing event time is not zero")
            elif type(observed) not in {int, float} or not math.isclose(
                observed, value, abs_tol=1e-8
            ):
                raise RunContractError(f"contract 1.0: inconsistent temporal offset {key}")
        if (
            time(offsets["event_time_utc"]) if offsets.get("event_time_utc") else None
        ) != event_time:
            raise RunContractError("contract 1.0: event-time offset mismatch")
        if event_time and (max(pre) >= event_time or post <= event_time):
            raise RunContractError("contract 1.0: acquisition/event-time order invalid")
        osm = manifest["osm"]
        snapshot = time(osm["snapshot"])
        limit = boundary.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        if snapshot > limit or (event_time and snapshot >= event_time):
            raise RunContractError("contract 1.0: OSM snapshot after allowed cutoff")
        proof = osm.get("cutoff_proof")
        if not isinstance(proof, dict) or time(proof.get("snapshot")) != snapshot:
            raise RunContractError("contract 1.0: OSM cutoff proof missing/inconsistent")
        missing = proof.get("features_without_timestamp")
        checked = proof.get("features_checked")
        if (
            type(missing) is not int
            or type(checked) is not int
            or not 0 <= missing <= checked
            or type(proof.get("passed")) is not bool
        ):
            raise RunContractError("contract 1.0: invalid cutoff proof counts")
        if proof["passed"] != (missing == 0) or (not report["synthetic"] and not proof["passed"]):
            raise RunContractError("contract 1.0: unavailable real OSM cutoff proof")
        if proof.get("max_edit_timestamp") and time(proof["max_edit_timestamp"]) > snapshot:
            raise RunContractError("contract 1.0: post-cutoff OSM edit")
        pairing = manifest.get("pairing")
        if not isinstance(pairing, dict):
            raise RunContractError("contract 1.0: pairing missing")
        if any(type(pairing.get(k)) is not bool for k in ("cross_platform", "platform_unknown")):
            raise RunContractError("contract 1.0: pairing flags must be bool")
        platforms = pairing.get("platforms")
        if (
            not isinstance(platforms, list)
            or len(platforms) != len(pre) + 1
            or any(s not in {"S1A", "S1B", "S1C", "S1D", "unknown"} for s in platforms)
        ):
            raise RunContractError("contract 1.0: invalid pairing platforms")
        if pairing.get("cross_platform") != (len(set(platforms) - {"unknown"}) > 1) or pairing.get(
            "platform_unknown"
        ) != ("unknown" in platforms):
            raise RunContractError("contract 1.0: inconsistent pairing flags")
        segmentation = manifest.get("road_segmentation")
        if not isinstance(segmentation, dict) or not isinstance(
            segmentation.get("ways_with_dropped_degenerate_intervals"), list
        ):
            raise RunContractError("contract 1.0: segmentation metadata missing")
        for key in ("max_segment_length_m", "degenerate_tolerance_m"):
            value: Any = segmentation.get(key)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise RunContractError("contract 1.0: invalid segment tolerance")
    elif offsets is not None and not isinstance(offsets, dict):
        raise RunContractError("contract 1.0: incomplete offsets must be object or null")
    return {
        "producing_commit": None if producer == "unknown" else producer,
        "producer_tree_dirty": dirty,
        "producer_commit_source": origin,
        "package_version": report["package_version"],
        "event_time": event_time,
        "event_date_boundary": boundary,
        "temporal_offsets": offsets,
    }


def validate_properties(layer: str, props: dict) -> None:
    if layer == "destinations" and props.get("destination_tier") not in TIERS:
        raise RunContractError("contract 1.0: destination_tier missing/unknown")
    if layer == "settlements":
        coverage = props.get("primary_care_coverage")
        if coverage not in {"missing_or_unsnapped_primary_care", "mapped_primary_care_available"}:
            raise RunContractError("contract 1.0: primary-care coverage missing/unknown")
        for key in CARE_FIELDS:
            if key not in props or (props[key] is not None and type(props[key]) is not bool):
                raise RunContractError("contract 1.0: primary-care access must be bool or null")
            if (props[key] is None) != (coverage == "missing_or_unsnapped_primary_care"):
                raise RunContractError("contract 1.0: primary-care unknown coverage mismatch")


def verify_osm_proof(root, manifest, settings) -> None:
    """Recompute the producer proof from the checksummed bundled OSM inputs."""
    import json

    from backend.app.upstream.paths import authorised_path

    osm = manifest["osm"]
    proof = osm["cutoff_proof"]
    checked = missing = 0
    latest = None
    fields = set()
    for layer in ("roads", "buildings", "settlements", "destinations"):
        if layer not in osm:
            continue
        path = authorised_path(root / osm[layer], settings, root=root)
        obj = json.loads(path.read_text())
        for feature in obj["features"]:
            props = feature["properties"]
            checked += 1
            field = next((k for k in ("edit_timestamp", "@lastEdit") if props.get(k)), None)
            if field is None:
                missing += 1
                continue
            fields.add(field)
            stamp = time(props[field])
            if stamp > time(osm["snapshot"]):
                raise RunContractError("contract 1.0: bundled OSM has post-cutoff edit")
            latest = stamp if latest is None else max(latest, stamp)
    recorded = time(proof["max_edit_timestamp"]) if proof.get("max_edit_timestamp") else None
    if (
        checked != proof["features_checked"]
        or missing != proof["features_without_timestamp"]
        or latest != recorded
        or proof.get("timestamp_field") != (",".join(sorted(fields)) or None)
    ):
        raise RunContractError("contract 1.0: OSM cutoff proof differs from bundled inputs")
