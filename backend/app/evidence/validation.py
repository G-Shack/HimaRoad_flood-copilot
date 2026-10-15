"""Evidence integrity checks, distinct from external model evaluation metrics."""

from datetime import UTC, datetime
from typing import Literal

from backend.app.evidence.claims import build_claims, read_json, run_id, verify_claim
from backend.app.evidence.vocabulary import map_status
from backend.app.schemas import Contract, EvidenceClaim, HimaRoadRun
from backend.app.settings import Settings


class ValidationIssue(Contract):
    code: str
    severity: Literal["error", "warning"]
    detail: str


class EvidenceValidation(Contract):
    result: Literal["passed", "failed", "unavailable"]
    issues: tuple[ValidationIssue, ...]
    recomputed_counts: dict[str, int]
    scope: str = "Evidence integrity only; does not establish mapping accuracy or input compliance."


def validate_evidence(
    run: HimaRoadRun, claims: tuple[EvidenceClaim, ...], settings: Settings
) -> EvidenceValidation:
    issues: list[ValidationIssue] = []

    def issue(code: str, detail: str, severity: Literal["error", "warning"] = "error"):
        issues.append(ValidationIssue(code=code, severity=severity, detail=detail))

    identifier = run_id(run)
    if len({c.id for c in claims}) != len(claims):
        issue("duplicate_claim", "Repeated claim IDs")
    known_ids = {c.id for c in claims}
    expected_ids = {c.id for c in build_claims(run, settings)}
    if not expected_ids.issubset(known_ids):
        issue(
            "source_facts", "Source facts are missing or differ from deterministic reconstruction"
        )
    for claim in claims:
        try:
            verify_claim(claim)
        except ValueError:
            issue("claim_hash", "Claim content differs from its ID")
        if claim.run_id != identifier:
            issue("run_identity", "Claim belongs to another run")
        if claim.parameters.get("producer_commit") != run.producing_commit:
            issue("producer_identity", "Claim producer differs from recorded producer")
        for ref in claim.sources:
            if (
                run.files.get(ref.file) != ref.sha256
                or ref.file in {"metrics.json", "comparison.md", "reference.json"}
                or "reference" in ref.file.split("/")
            ):
                issue("source_provenance", "Claim source is not a permitted checksummed run output")
        if claim.scenario not in {"observation", "baseline"}:
            if claim.scenario not in run.scenarios or "hypothetical" not in claim.quality_flags:
                issue("scenario_label", "Scenario is unreviewed or not labelled hypothetical")
        if any(parent not in known_ids for parent in claim.parameters.get("derived_from", [])):
            issue("derivation", "Missing parent claim")
        if run.synthetic and "synthetic" not in claim.quality_flags:
            issue("synthetic_label", "Synthetic evidence lacks its label")
        for name, synthetic in (run.component_synthetic or {}).items():
            if synthetic is True and f"synthetic_{name}" not in claim.quality_flags:
                issue("synthetic_component_label", "Synthetic component evidence lacks its label")
    if run.producing_commit is None:
        issue(
            "producer_unknown",
            "Producer commit is unknown; adapter pin is not producer identity",
            "warning",
        )
    if run.status == "incomplete":
        if any(
            c.kind in {"summary", "area", "feature", "route", "repair_priority"} for c in claims
        ):
            issue("incomplete_facts", "Incomplete run contains impact facts")
        return EvidenceValidation(
            result="failed" if any(i.severity == "error" for i in issues) else "unavailable",
            issues=tuple(issues),
            recomputed_counts={},
        )
    layers = {
        layer: read_json(run, settings, layer + ".geojson")["features"]
        for layer in ("roads", "buildings", "settlements")
    }
    for layer, features in layers.items():
        ids = [f["properties"]["id"] for f in features]
        if len(ids) != len(set(ids)):
            issue("duplicate_feature", f"{layer} contains repeated feature IDs")
        subjects = {f"{layer}:{f['properties']['id']}": f["properties"] for f in features}
        feature_claims = {c.subject: c for c in claims if c.kind == "feature" and c.label == layer}
        if set(feature_claims) != set(subjects):
            issue("feature_completeness", f"{layer} feature claims missing or extra")
        for subject, props in subjects.items():
            if subject not in feature_claims:
                continue
            claim = feature_claims[subject]
            expected = map_status(
                "settlement" if layer == "settlements" else "evidence",
                props["category" if layer == "settlements" else "evidence_category"],
            )
            if (
                claim.status != expected
                or claim.value["properties"].get("status") != expected.value
            ):
                issue(
                    "status_mapping", f"{subject} status differs from conservative source mapping"
                )
    settlements = [f["properties"] for f in layers["settlements"]]
    buildings = [f["properties"] for f in layers["buildings"]]
    roads = [f["properties"] for f in layers["roads"]]
    counts = {
        "potentially_road_cut": sum(p["potentially_isolated"] for p in settlements),
        "access_uncertain": sum(
            p["depends_only_on_unknown"]
            if run.schema_version == "3.0"
            else p["baseline_accessible"]
            and p["scenario2_accessible"]
            and not p["scenario3_accessible"]
            for p in settlements
        ),
        "trail_access_possible": sum(
            p["potentially_isolated"] and p["pedestrian_accessible"] for p in settlements
        ),
        "potentially_exposed_buildings": sum(p["potentially_exposed"] for p in buildings),
        "suspected_affected_links": sum(
            p["evidence_category"] in {"likely_impact", "observed_change", "possible_impact"}
            for p in roads
        ),
    }
    if run.schema_version in {"1.0", "3.0"}:
        counts["primary_care_potentially_cut"] = sum(
            p["primary_care_baseline_accessible"] is True
            and p["primary_care_scenario2_accessible"] is False
            for p in settlements
        )
    if run.schema_version == "3.0":
        counts.update(
            corroborated_road_cut=sum(
                p["isolation_confidence"] == "corroborated" for p in settlements
            ),
            depends_only_on_unknown=sum(p["depends_only_on_unknown"] for p in settlements),
            settlements_total=len(settlements),
            settlements_from_building_clusters=sum(
                p["kind"] == "building_cluster" for p in settlements
            ),
        )
    for key, value in counts.items():
        if run.summary is None or run.summary.get(key) != value:
            issue(
                "summary_mismatch",
                f"{key}: recomputed {value}; "
                f"producer {run.summary.get(key) if run.summary else None}",
            )
    for p in settlements:
        if p["potentially_isolated"] and (
            not p["baseline_accessible"] or p["scenario2_accessible"]
        ):
            issue(
                "baseline_upgrade",
                f"{p['id']}: newly disconnected flag contradicts baseline/scenario",
            )
    manifest = read_json(run, settings, "manifest.json")
    boundary = run.event_date_boundary or datetime.combine(
        run.event_date, datetime.min.time(), tzinfo=UTC
    )

    def timestamp(text: str) -> datetime:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))

    if any(timestamp(p["timestamp"]) >= boundary for p in manifest["pre"]):
        issue("temporal_order", "Pre observation does not precede supplied date boundary")
    if timestamp(manifest["post"]["timestamp"]) <= boundary:
        issue("temporal_order", "Post observation does not follow supplied date boundary")
    if timestamp(manifest["osm"]["snapshot"]) >= boundary:
        issue("osm_cutoff", "OSM snapshot is not before supplied event date")
    return EvidenceValidation(
        result="failed" if any(i.severity == "error" for i in issues) else "passed",
        issues=tuple(issues),
        recomputed_counts=counts,
    )
